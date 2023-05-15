from typing import Generator
from ..db.engine import get_engine
from ..config.logger import get_logger

logger = get_logger()

def get_connection(settings) -> Generator:
    logger.info("Initializing connection with db")
    engine = get_engine(settings)
    conn = engine.connect()
    try:
        yield conn
    finally:
        logger.debug("Closing connection")
        conn.close()
