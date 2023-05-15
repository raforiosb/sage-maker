from typing import Generator
from ..config.settings import settings
from ..db.engine import get_engine
from ..config.logger import logger

import redis

from koombea_serve.connection.database_tunnel import get_tunnel


def get_connection() -> Generator:
    logger.info("Initializing connection with db")
    tunnel = None
    conn = None
    try:
        tunnel = get_tunnel()
        tunnel.start()
        engine = get_engine(str(tunnel.local_bind_port))
        conn = engine.connect()
        yield conn
    finally:
        logger.debug("Closing connection")
        conn.close()
        tunnel.close()


def get_redis_connection() -> Generator:
    logger.info("Initializing connection to redis db")
    conn = None
    try:
        conn = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_PORT_DB,
            )
        yield conn
    finally:
        logger.info("Closing connection")
        conn.close()
