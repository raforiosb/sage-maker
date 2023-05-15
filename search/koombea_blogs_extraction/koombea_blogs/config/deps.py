import wandb
from wandb.wandb_run import Run


def get_wandb_run_extraction(settings) -> Run:
    run = wandb.init(
        project=settings.WANDDB_PROJECT_NAME,
        entity=settings.WANDB_ENTITY,
        tags=[settings.STAGE, "ETL-tracking"],
        name="data-extraction-job-" + settings.MYSQL_DBNAME,
        job_type="data-extraction-job",
        notes="Data extraction job to keep track of koombea blogs",
        config={
            "project_name": settings.WANDDB_PROJECT_NAME,
            "entity": settings.WANDB_ENTITY,
            "db_stage": settings.STAGE,
            "db_url": settings.SQLALCHEMY_DATABASE_URL,
        },
        id="data-extraction-job-" + settings.MYSQL_DBNAME,
        mode=settings.WANDB_MODE,
        resume=True,
    )
    return run


def get_wandb_run_preprocessing(settings) -> Run:
    run = wandb.init(
        project=settings.WANDDB_PROJECT_NAME,
        entity=settings.WANDB_ENTITY,
        tags=[settings.STAGE, "ETL-tracking"],
        name="data-processing-job" + "-" + settings.MYSQL_DBNAME,
        job_type="data-processing-job",
        notes="Data processing and clean job to keep track of koombea blogs",
        config={
            "project_name": settings.WANDDB_PROJECT_NAME,
            "entity": settings.WANDB_ENTITY,
            "db_stage": settings.STAGE,
            "db_url": settings.SQLALCHEMY_DATABASE_URL,
        },
        id="data-processing-job" + "-" + settings.MYSQL_DBNAME,
        mode=settings.WANDB_MODE,
        resume=True,
    )
    return run


def get_wandb_run_visualizing_analysis(settings) -> Run:
    run = wandb.init(
        project=settings.WANDDB_PROJECT_NAME,
        entity=settings.WANDB_ENTITY,
        tags=[settings.STAGE, "ETL-tracking"],
        name="data-visualizing-analysis-job" + "-" + settings.MYSQL_DBNAME,
        job_type="data-visualizing-job",
        notes="Data visualizing analysis job to keep track of koombea blogs",
        config={
            "project_name": settings.WANDDB_PROJECT_NAME,
            "entity": settings.WANDB_ENTITY,
            "db_stage": settings.STAGE,
            "db_url": settings.SQLALCHEMY_DATABASE_URL,
        },
        mode=settings.WANDB_MODE,
        id="data-visualizing-analysis-job" + "-" + settings.MYSQL_DBNAME,
        resume=True,
    )
    return run
