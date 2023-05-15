import os
import subprocess

from app.utils import logger
from celery import Celery
from celery.schedules import crontab

celery_app = Celery(broker='redis://localhost:6379/0')

script = "cache.py"

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        crontab(minute="0", hour="*/2"),
        execute_cache.s('Start Cronjob Cache!'),
        name='Load Data and Cache All Predictions'
    )
    
@celery_app.task
def execute_cache(args):
    """Test Celery"""
    logger.info(args)
    
    # Get commands for online training
    train_cmd = [script]
    # logger args
    logger.info(train_cmd)
    # Execute subprocess 
    process = subprocess.Popen(
        ' '.join(train_cmd),
        shell = True,
        stderr = subprocess.STDOUT,
        env = os.environ
    )
    # Wait until finish running the training scripts
    stdout, stderr = process.communicate()
    return_code = process.poll()
    if return_code:
        logger.error("There was an error on the training script")
        error_msg = 'Return code: {}, CMD: {}'.format(return_code, train_cmd)
        raise Exception(error_msg)
    else:
        logger.info("Training script run succesfully")
