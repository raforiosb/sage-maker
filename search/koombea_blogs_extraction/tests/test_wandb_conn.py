import wandb
from wandb.wandb_run import Run
import pandas as pd
from koombea_blogs.config.settings import settings


def test_log_table2wandb(wandb_run: Run):
    blogs_df = pd.read_csv(f"test_data/{settings.DB_NAME}_blogs.csv")
    log_artifact_success = True
    message = "Wandb cannot upload the artifact"
    try:
        artifact = wandb.Artifact(
            f"test_blogs_{settings.DB_NAME}",
            type="dataset",
            description="Test dataset artifact",
            metadata={
                "db": settings.DB_NAME,
                "stage": settings.STAGE,
                "all_good": True,
            },
        )
        artifact.add(wandb.Table(dataframe=blogs_df), name="blogs_dataframe")
        wandb_run.log_artifact(artifact)
        # wandb_run.log_artifact(wandb.Table(dataframe=blogs_df))
    except Exception as error:
        log_artifact_success = False
        message += " " + str(error)
    assert log_artifact_success, message
