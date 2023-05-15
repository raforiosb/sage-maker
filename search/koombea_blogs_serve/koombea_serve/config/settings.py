from pydantic import BaseSettings, BaseModel, validator, AnyUrl
from typing import Optional, Any, Dict
import logging, os

class MySqlDsn(AnyUrl):
    allowed_schemes = {"mysql+pymysql"}
    user_required = True


class Settings(BaseSettings):
    LOG_LEVEL: int = os.environ.get(
        "SM_LOG_LEVEL", logging.DEBUG
    )  # get log level for sagemaker
    PEM_FILE: str
    HOSTNAME: str
    USERNAME: str
    PASSWORD: str
    MYSQL_DBNAME: str
    MYSQL_HOSTNAME: str
    SSH_PORT: str
    MYSQL_PORT: str
    STAGE: str
    
    def assemble_db_url_connection(
        self, tunnel_port: str
    ) -> str:
        return MySqlDsn.build(
            scheme="mysql+pymysql",
            user=self.USERNAME,
            password=self.PASSWORD,
            port=tunnel_port,
            host=self.MYSQL_HOSTNAME,
            path=f"/{self.MYSQL_DBNAME}",
        )

    # wandb configuration
    WANDB_API_KEY: str
    WANDDB_PROJECT_NAME: str
    WANDB_ENTITY: str
    WANDB_MODE: str

    # celery config
    CRON_MINUTE: str = "0"
    CRON_HOUR: str = "*/2"
    CRON_BROKER_NAME: str = "redis"

    # redis broker
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_PORT_DB: str = "0"

    CRON_BROKER_URL: Optional[str] = None

    @validator("CRON_BROKER_URL")
    def assemble_cron_broker_url_connection(
        cls, v: Optional[str], values: Dict[str, Any]
    ) -> str:
        return "{}://{}:{}/{}".format(
            values["CRON_BROKER_NAME"],
            values["REDIS_HOST"],
            values["REDIS_PORT"],
            values["REDIS_PORT_DB"],
        )

    # cache script
    CACHE_SCRIPT: str = "cache.py"

    # model artifacts
    SM_MODEL_DIR: str = "/opt/ml/model"


settings = Settings()
