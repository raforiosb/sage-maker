import logging
from ..config.settings import settings
import traceback

BASIC_FORMAT_LOGGING = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=BASIC_FORMAT_LOGGING, level=settings.LOG_LEVEL)
logger = logging.getLogger()


def logger_error(error):
    logger.error("error: {}".format(error))
    trc = traceback.format_exc()
    logger.error(trc)
