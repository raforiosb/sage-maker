from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_CONFIG

Base = declarative_base()

def get_session_and_engine(tunnel=None, tunnel_prod = None):
    engine, SessionLocal = None, None
    if tunnel is not None:
        SQLALCHEMY_DATABASE_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(DATABASE_CONFIG['USERNAME'],
                                                                DATABASE_CONFIG['PASSWORD'],
                                                                DATABASE_CONFIG['MYSQL_HOSTNAME'],
                                                                tunnel.local_bind_port,
                                                                DATABASE_CONFIG['MYSQL_DBNAME'])

        engine = create_engine(
            SQLALCHEMY_DATABASE_URL,
            pool_pre_ping=True
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    engine_prod, SessionLocalProd = None, None
    if DATABASE_CONFIG.get("DEV", False) and tunnel_prod is not None:
        SQLALCHEMY_DATABASE_URL_PROD = "mysql+pymysql://{}:{}@{}:{}/{}".format(DATABASE_CONFIG['PROD_USERNAME'],
                                                                DATABASE_CONFIG['PROD_PASSWORD'],
                                                                DATABASE_CONFIG['MYSQL_HOSTNAME'],
                                                                tunnel_prod.local_bind_port,
                                                                DATABASE_CONFIG['PROD_MYSQL_DBNAME'])

        engine_prod = create_engine(
            SQLALCHEMY_DATABASE_URL_PROD,
            pool_pre_ping=True
        )

        SessionLocalProd = sessionmaker(autocommit=False, autoflush=False, bind=engine_prod)
        
    return [
        {"session": SessionLocal, "engine": engine},
        {"session": SessionLocalProd, "engine": engine_prod}
    ]
