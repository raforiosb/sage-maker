import json
import boto3
from sagemaker import predictor
from sagemaker.session import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def lambda_handler(event, context):
    # get basic configuration
    sagemaker_session = Session()
    sagemaker_client = sagemaker_session.sagemaker_client

    # stages
    print(event, vars(context))

    for record in event["Records"]:
        try:
            payload = json.loads(record["body"])
            token = payload["token"]
            arguments = payload["arguments"]

            if arguments["stage"] == "staging":
                endpoint_name = "blogsearch-stage-dev-2020-08-11-18-01-30"  # dev
            elif arguments["stage"] == "prod":
                endpoint_name = "blogsearch-stage-prod-2020-08-11-22-52-22"  # prod

            print(arguments)
            # retrieve model name from arguments
            model_name = arguments["model_name"]
            # endpoint_name = "blogsearch-stage-dev-2020-08-11-18-01-30"  # dev

            # initialize predictor
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=sagemaker_session,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
            )

            # update predictor
            predictor.update_endpoint(
                initial_instance_count=arguments["initial_instance_count"],
                instance_type=arguments["instance_type"],
                model_name=model_name,
                wait=True,
            )

            data = {
                "s": "difference between web and mobile apps",
                "lang": "en",
                "per_page": 2,
                "page": 1,
                # "term":["hi-tech", "iot"]
            }

            prediction = predictor.predict(data)

            pipeline_success_info = (
                sagemaker_client.send_pipeline_execution_step_success(
                    CallbackToken=token,
                    OutputParameters=[{"Name": "status", "Value": "ok"}],
                )
            )

            print(
                "succes: {} with pipeline success info {} \ndata\n{}:\nprediction:\n{}".format(
                    "Todo OK", pipeline_success_info, data, prediction
                )
            )
        except Exception as error:
            pipeline_failure_info = (
                sagemaker_client.send_pipeline_execution_step_failure(
                    CallbackToken=token, FailureReason=str(error)
                )
            )
            print(
                "failure: {} with pipeline failure info: {}".format(
                    error, pipeline_failure_info
                )
            )


if __name__ == "__main__":
    lambda_handler("", "")
