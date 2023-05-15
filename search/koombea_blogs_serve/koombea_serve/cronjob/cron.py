import os
import subprocess

from koombea_serve.config.logger import logger
from koombea_serve.config.settings import settings

from celery import Celery
from celery.schedules import crontab


celery_app = Celery(broker=settings.CRON_BROKER_URL)


@celery_app.task
def execute_cache(args):
    logger.info(args)

    # get commands for extract and cache vectors
    cmd = [settings.CACHE_SCRIPT]
    logger.debug("commands: {}".format(cmd))

    # execute subprocess
    cache_process = subprocess.Popen(
        " ".join(cmd), shell=True, stderr=subprocess.STDOUT, env=os.environ
    )
    # wait until finish runnting the cache script
    stdout, stderr = cache_process.communicate()
    code = cache_process.poll()
    if code:
        logger.error("There was an error on the cache script")
        error_msg = "return code: {}, cmd: {}, stdout: {}, stderr: {}".format(
            code, cmd, stdout, stderr
        )
        raise Exception(error_msg)
    logger.info("Cache script run succesfully!")


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        # crontab(minute=settings.CRON_MINUTE, hour=settings.CRON_HOUR),
        crontab(minute=settings.CRON_MINUTE, hour=settings.CRON_HOUR),
        execute_cache.s("Start cronjob cache!"),
        name="Load Data and Cache all vectors",
    )
