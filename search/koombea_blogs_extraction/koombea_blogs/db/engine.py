from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Create session and engine for sqlalchemy
def get_engine(settings) :
    engine: Engine = create_engine(url=settings.SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
    return engine
