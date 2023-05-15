import wandb
from .config.logger import logger, get_wandb_run_training
from .config.settings import settings
from pathlib import Path
import pandas as pd
import json


class Data:
    def __init__(
        self,
        blogs_df_path: Path,
        en_data_path: Path,
        es_data_path: Path,
        model_dir: Path,
    ):
        logger.info("Loading data files for training and model analysis")
        self.blogs_df = pd.read_csv(blogs_df_path)
        self.model_dir = model_dir

        self.en_data = json.load(open(en_data_path, "r"))
        self.en_idx2ids, _ = zip(*self.en_data)
        self.en_num_blogs = len(self.en_data)

        self.es_data = json.load(open(es_data_path, "r"))
        self.es_idx2ids, _ = zip(*self.es_data)
        self.es_num_blogs = len(self.es_data)

    def save(self):
        # save mappings as an artifact
        ids_mapping_en_name = self.model_dir / "idx2ids_mapping_en_{}.json".format(
            settings.MYSQL_DBNAME
        )
        ids_mapping_es_name = self.model_dir / "idx2ids_mapping_es_{}.json".format(
            settings.MYSQL_DBNAME
        )

        with open(ids_mapping_en_name, "w") as ids_en_file, open(
            ids_mapping_es_name, "w"
        ) as ids_es_file:
            json.dump(self.en_idx2ids, ids_en_file)
            json.dump(self.es_idx2ids, ids_es_file)

        # Save blogs df as an artifact
        blogs_df_name = self.model_dir / "blogs_df_{}.csv".format(settings.MYSQL_DBNAME)
        self.blogs_df.to_csv(blogs_df_name, index=False)

    def build_metadata(self):
        run = get_wandb_run_training()
        artifact = wandb.Artifact(
            name="idx2ids_mappings",
            type="result-data",
            description="ids2idx mapping, this json file maps index matrix vector to blogs id",
            metadata={"db_stage": settings.STAGE, "db_name": settings.MYSQL_DBNAME},
        )
        ids_mapping_en_name = self.model_dir / "idx2ids_mapping_en_{}.json".format(
            settings.MYSQL_DBNAME
        )
        ids_mapping_es_name = self.model_dir / "idx2ids_mapping_es_{}.json".format(
            settings.MYSQL_DBNAME
        )
        blogs_df_name = self.model_dir / "blogs_df_{}.csv".format(settings.MYSQL_DBNAME)
        # add and log artifacts
        artifact.add_file(ids_mapping_en_name)
        artifact.add_file(ids_mapping_es_name)
        artifact.add_file(blogs_df_name)
        run.log_artifact(artifact)
