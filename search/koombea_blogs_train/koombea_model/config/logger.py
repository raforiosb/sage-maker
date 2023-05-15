import logging
from ..config.settings import settings

BASIC_FORMAT_LOGGING = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=BASIC_FORMAT_LOGGING, level=settings.LOG_LEVEL)
logger = logging.getLogger()

import wandb
from wandb.wandb_run import Run


def get_wandb_run_training() -> Run:
    run = wandb.init(
        project=settings.WANDDB_PROJECT_NAME,
        entity=settings.WANDB_ENTITY,
        tags=[settings.STAGE, "Training-Tracking"],
        name="semantic-model-analisys-training-job-" + settings.MYSQL_DBNAME,
        job_type="training-job",
        notes="Training tracking job with analysis and very cool visualizations",
        config={
            "project_name": settings.WANDDB_PROJECT_NAME,
            "entity": settings.WANDB_ENTITY,
            "db_stage": settings.STAGE,
        },
        id="analysis-training-job-" + settings.MYSQL_DBNAME,
        mode=settings.WANDB_MODE,
        resume=True,
    )
    return run
