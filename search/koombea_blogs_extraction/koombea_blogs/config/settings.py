import logging
import os
from pydantic import BaseSettings, BaseModel, validator, AnyUrl
from typing import Optional, Any, Dict
from koombea_blogs.config.logger import get_logger

logger = get_logger()

class MySqlDsn(AnyUrl):
    allowed_schemes = {"mysql+pymysql"}
    user_required = True


class Settings(BaseSettings):
    LOG_LEVEL: int = logging.INFO
    PEM_FILE: str
    HOSTNAME: str
    USERNAME: str 
    PASSWORD: str 
    MYSQL_DBNAME: str
    MYSQL_HOSTNAME: str
    SSH_PORT: str 
    MYSQL_PORT: str
    TUNNEL_PORT: Optional[str] = None
    STAGE: str
    SQLALCHEMY_DATABASE_URL: Optional[str] = None
    
    @validator("PEM_FILE")
    def validate_pem_file(
        cls, v
    ) -> str:
        logger.info(v)
        if os.path.exists(v):
            return v
        else:
            raise ValueError(f"{v} path does not exists")


    def set_assemble_db_url_connection(
        self, tunnel_port
    ) -> str:
        self.TUNNEL_PORT = str(tunnel_port)
        self.SQLALCHEMY_DATABASE_URL=MySqlDsn.build(
            scheme="mysql+pymysql",
            user=self.USERNAME,
            password=self.PASSWORD,
            host=self.MYSQL_HOSTNAME,
            port=self.TUNNEL_PORT,
            path=f"/{self.MYSQL_DBNAME}",
        )
        
    # wandb configuration
    WANDB_API_KEY: str
    WANDDB_PROJECT_NAME: str 
    WANDB_ENTITY: str 
    WANDB_MODE: str 

def get_settings() :
    settings = Settings()
    return settings
