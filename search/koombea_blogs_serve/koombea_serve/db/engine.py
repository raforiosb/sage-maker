from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from ..config.settings import settings

# Create session and engine for sqlalchemy
def get_engine(tunnel_port: str):
    engine: Engine = create_engine(url=settings.assemble_db_url_connection(tunnel_port), pool_pre_ping=True)
    return engine