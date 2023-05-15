import logging
def get_logger(settings=None):
    BASIC_FORMAT_LOGGING = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=BASIC_FORMAT_LOGGING, level=settings.LOG_LEVEL if settings else logging.INFO)
    logger = logging.getLogger()
    return logger
