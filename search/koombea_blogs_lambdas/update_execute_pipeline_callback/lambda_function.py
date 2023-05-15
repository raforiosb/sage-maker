from sagemaker.session import Session
from sagemaker.processing import Processor, ProcessingOutput
from sagemaker.network import NetworkConfig
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import CreateModelStep
from sagemaker.workflow.callback_step import (
    CallbackStep,
    CallbackOutput,
    CallbackOutputTypeEnum,
)
from sagemaker.model import Model
from time import gmtime, strftime

import pprint
import json


def get_execution_role(sagemaker_session: Session):
    role = sagemaker_session.boto_session.client("iam").get_role(
        RoleName="AmazonSageMaker-ExecutionRole-20230105T181131"
    )["Role"]["Arn"]
    return role


def load_env_variables(*env_files):
    env_vars = dict()
    get_values = lambda x: [(x.split("=")[0], x.split("=")[1])]
    for env_file in env_files:
        with open(env_file, "r") as file:
            env_vars.update(
                dict(
                    [
                        (key.strip(), value.strip())
                        for line in file.readlines()
                        for key, value in get_values(line)
                    ]
                )
            )
    return env_vars


def get_configurations(stage="staging"):
    environment = load_env_variables("vars.env", f"vars.{stage}.env")
    return environment


# I've already create a vpc configuration that is able to connect to koombea db
def get_koombea_db_vpc_conf(ec2_client, account_id, security_group_name):
    # Get subnets
    subnets = ec2_client.describe_subnets(
        Filters=[
            {
                'Name':'owner-id',
                'Values':[account_id]
            }
        ]
    )
    # choose just the private subnets routing to the NateGateway
    subnets_ids = [subnets_["SubnetId"]
                   for subnets_ in subnets["Subnets"]
                   if "Tags" in subnets_.keys() and 'sm' == subnets_["Tags"][0]["Value"].split("-")[0] and "p" in subnets_["Tags"][0]["Value"]]
    # get security groups
    security_groups = ec2_client.describe_security_groups(
        Filters=[
            {
                "Name":"owner-id",
                "Values":[account_id]
            },
            {
                "Name":"group-name",
                "Values":[security_group_name]
            }
        ]
    )
    sec_groups_ids = [sec_groups_["GroupId"] for sec_groups_ in security_groups["SecurityGroups"]]
    return {"Subnets":subnets_ids,
            "SecurityGroupIds":sec_groups_ids}

def get_repository_info(ecr_client, repository_name, account_id):
    repository_info = ecr_client.describe_repositories(
        registryId=account_id, repositoryNames=[repository_name]
    )["repositories"][0]
    return repository_info


def create_extraction_processing_step(stage: str, **kwargs):
    # processor config
    environment = get_configurations(stage)
    network_config = NetworkConfig(
        security_group_ids=kwargs["vpc_config"]["SecurityGroupIds"],
        subnets=kwargs["vpc_config"]["Subnets"],
    )
    # process output config
    source_output = "/opt/ml/processing/processed_data"
    s3_processor_output_bucket = "s3://{}/{}/{}".format(
        kwargs["default_bucket"], "koombea_website_ml", "koombea_blogs_information"
    )
    # define processor job
    blogs_processor = Processor(
        role=kwargs["role"],
        image_uri=kwargs["processor_image_uri"],
        instance_count=1,
        instance_type="ml.t3.large",
        entrypoint=["python", "run.py"],
        base_job_name="etl-koombea-blogs-job",
        sagemaker_session=kwargs["sagemaker_session"],
        env=environment,
        network_config=network_config,
    )
    # define step process
    step_process = ProcessingStep(
        name="BlogsExtractionProcessingJob{}".format(stage.capitalize()),
        processor=blogs_processor,
        outputs=[
            ProcessingOutput(
                output_name="train",
                source=source_output,
                destination=s3_processor_output_bucket,
            )
        ],
        job_arguments=["--output-path", source_output],
    )
    return step_process


def create_train_step(stage: str, step_process: ProcessingStep, **kwargs):
    # train config
    environment = get_configurations(stage)
    # s3_train_input_bucket = kwargs["s3_processor_output_bucket"]
    s3_train_output_bucket = "s3://{}/{}/{}".format(
        kwargs["default_bucket"], "koombea_website_ml", "koombea_blogs_models"
    )
    hyperparameters = {
        "min_count": 0,
        "size": 300,
        "sg": 1,
        "window": 15,
        "iter": 40,
        "sample": 6e-5,
        "hs": 0,
        "negative": 15,
        "ns_exponent": -0.5,
    }
    # define estimator
    blogs_train = Estimator(
        image_uri=kwargs["trainer_image_uri"],
        role=kwargs["role"],
        instance_count=1,
        instance_type="ml.m5.large",
        base_job_name="koombea-blogs-vector-train",
        sagemaker_session=kwargs["sagemaker_session"],
        hyperparameters=hyperparameters,
        output_path=s3_train_output_bucket,
        environment=environment,
    )
    # define step train
    step_train = TrainingStep(
        name="BlogsTrainingJob{}".format(stage.capitalize()),
        estimator=blogs_train,
        inputs={
            "training": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri
            )
        },
    )
    return step_train


def register_model_for_deploy(stage: str, step_train: TrainingStep, **kwargs):
    # model config
    environment = get_configurations(stage)

    # create model
    blogs_model = Model(
        image_uri=kwargs["serve_image_uri"],
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=kwargs["role"],
        env=environment,
        vpc_config=kwargs["vpc_config"],
        sagemaker_session=kwargs["sagemaker_session"],
    )
    # create step model
    create_inputs = CreateModelInput(instance_type="ml.t2.medium")
    step_create = CreateModelStep(
        name="CreateBlogsModel{}".format(stage.capitalize()),
        model=blogs_model,
        inputs=create_inputs,
    )
    return step_create


def create_deploy_step(stage: str, step_create: CreateModelStep, sqs_queue_url: str):
    step_callback = CallbackStep(
        name="DeployCallback{}".format(stage.capitalize()),
        sqs_queue_url=sqs_queue_url,
        inputs={
            "stage": stage,
            "model_name": step_create.properties.ModelName,
            "instance_type": step_create.inputs.instance_type,
            "initial_instance_count": 1,
        },
        outputs=[
            CallbackOutput(
                output_name="status", output_type=CallbackOutputTypeEnum.String
            )
        ],
        depends_on=[step_create.name],
    )
    return step_callback


def create_pipeline(
    stage,
    sagemaker_session,
    role,
    default_bucket,
    vpc_config,
    processor_image_uri,
    trainer_image_uri,
    serve_image_uri,
):
    print("Creating step process for stage : {}".format(stage))
    step_process = create_extraction_processing_step(
        stage,
        sagemaker_session=sagemaker_session,
        role=role,
        default_bucket=default_bucket,
        vpc_config=vpc_config,
        processor_image_uri=processor_image_uri,
    )

    print("Creating training step for stage : {}".format(stage))
    step_train = create_train_step(
        stage,
        step_process,
        sagemaker_session=sagemaker_session,
        role=role,
        default_bucket=default_bucket,
        trainer_image_uri=trainer_image_uri,
    )

    print("Creating a model create step process for stage: {}".format(stage))
    step_create = register_model_for_deploy(
        stage,
        step_train,
        sagemaker_session=sagemaker_session,
        role=role,
        vpc_config=vpc_config,
        serve_image_uri=serve_image_uri,
    )

    print("Creating deploy callback step for stage: {}".format(stage))
    step_deploy = create_deploy_step(
        stage,
        step_create,
        sqs_queue_url="https://sqs.us-west-2.amazonaws.com/256305374409/deploy_pipeline_callback_step",
    )

    # create pipeline
    pipeline = Pipeline(
        name=f"koombea-blogs-pipeline-{stage}",
        steps=[step_process, step_train, step_create, step_deploy],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def get_general_configuration():
    # general config
    sagemaker_session = Session()
    region = sagemaker_session.boto_region_name
    default_bucket = sagemaker_session.default_bucket()
    role = get_execution_role(sagemaker_session)
    account_id = sagemaker_session.account_id()
    vpc_config = get_koombea_db_vpc_conf(
        ec2_client=sagemaker_session.boto_session.client("ec2"),
        account_id=account_id,
        security_group_name="launch-wizard-1",
    )
    return {
        "sagemaker_session": sagemaker_session,
        "account_id": account_id,
        "region": region,
        "role": role,
        "vpc_config": vpc_config,
        "default_bucket": default_bucket,
    }


def lambda_handler(event, context):
    print("Create or update pipeline")
    config = get_general_configuration()
    # get ecr
    print("get ecr configuration")
    ecr_client = config["sagemaker_session"].boto_session.client("ecr")
    processor_image_uri = get_repository_info(
        ecr_client=ecr_client,
        repository_name="koombea-blogs-extraction-component",
        account_id=config["account_id"],
    )["repositoryUri"]
    trainer_image_uri = get_repository_info(
        ecr_client=ecr_client,
        repository_name="koombea-blogs-train-component",
        account_id=config["account_id"],
    )["repositoryUri"]
    serve_image_uri = get_repository_info(
        ecr_client=ecr_client,
        repository_name="koombea-blogs-serve-component",
        account_id=config["account_id"],
    )["repositoryUri"]
    # define stages
    stages = ["staging", "prod"]
    pipelines_stages = {}

    print("Stages: {}".format(stages))
    for stage in stages:
        print("update/create and execute pipeline step for stage: {}".format(stage))
        # create pipeline
        pipeline = create_pipeline(
            stage,
            config["sagemaker_session"],
            config["role"],
            config["default_bucket"],
            config["vpc_config"],
            processor_image_uri,
            trainer_image_uri,
            serve_image_uri,
        )

        pprint.pprint(pipeline.definition())
        pipelines_stages[pipeline.name] = pipeline

        print("Create pipeline for stage: {}".format(stage))
        service_dict = pipeline.upsert(role_arn=config["role"])
        pprint.pprint(service_dict)

        print("Execute pipeline for stage: {}".format(stage))
        execution = pipeline.start()

        print("Describe: {}".format(pipeline.name))
        description = execution.describe()
        pprint.pprint(description)

        print(
            "Success Status Code: {}".format(
                description["ResponseMetadata"]["HTTPStatusCode"]
            )
        )
        print("\n")
    # print pipelines
    print(pipelines_stages)


if __name__ == "__main__":
    lambda_handler("", "")
