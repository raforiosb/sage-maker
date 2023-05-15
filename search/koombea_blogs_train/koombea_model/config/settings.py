from typing import Any, Dict, Optional
from pydantic import BaseSettings, BaseModel, validator
from pydantic.fields import ModelField
from pathlib import Path
import os
import logging


class Settings(BaseSettings):
    LOG_LEVEL: int = os.environ.get("SM_LOG_LEVEL", logging.DEBUG)  # get log level for sagemaker
    MYSQL_DBNAME: str
    STAGE: str

    # wandb configuration
    WANDB_API_KEY: str
    WANDDB_PROJECT_NAME: str
    WANDB_ENTITY: str
    WANDB_MODE: str

    # filenames
    BLOGS_DF_NAME_FILE_CSV: Optional[str] = None
    EN_DATA_NAME_FILE_JSON: Optional[str] = None
    ES_DATA_NAME_FILE_JSON: Optional[str] = None

    @validator(
        "BLOGS_DF_NAME_FILE_CSV", "EN_DATA_NAME_FILE_JSON", "ES_DATA_NAME_FILE_JSON"
    )
    def validate_name_files(
        cls, v: Optional[str], values: Dict[str, Any], field: ModelField
    ) -> str:
        if not v:
            key_name_validating = field.alias
            options = key_name_validating.split("_")
            file_name = "_".join(map(lambda x: x.lower(), options[:2]))
            file_extension = options[-1].lower()
            db_name = values["MYSQL_DBNAME"]
            v = f"{file_name}_{db_name}.{file_extension}"
        return v


class PathSettings(BaseModel):
    blogs_df_path: Path
    en_data_path: Path
    es_data_path: Path
    model_dir: Path

    @validator(
        "blogs_df_path", "en_data_path", "es_data_path", "model_dir", allow_reuse=True
    )
    def validate_file_existence(cls, v: Path, field: ModelField) -> Path:
        if not v.exists():
            raise ValueError(
                f"path : {v} does not contains any file for {field.alias} argument"
            )
        return v


settings = Settings()
