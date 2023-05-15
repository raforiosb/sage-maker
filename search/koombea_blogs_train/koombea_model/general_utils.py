import json
import logging
import os
from typing import Any, Dict, Optional
from .config.logger import logger
from .config.settings import settings
from pydantic import ValidationError, BaseModel


def pretty_string(obj: Dict[str, Any]) -> str:
    obj = {str(key): str(value) for key, value in obj.items()}
    return json.dumps(obj, indent=True)


def parse_and_validate_arguments(
    Model: BaseModel, args: Dict[str, Any], name: Optional[str] = None
) -> BaseModel:
    try:
        return Model(**args)
    except ValidationError as error:
        logger.error(
            "There was an error validating arguments for {}, error: {}".format(
                name, error
            )
        )
        raise error


def get_channel_path(index: int) -> str:
    channel_name: str = json.loads(os.environ.get("SM_CHANNELS"))[index]
    return os.environ.get("SM_CHANNEL_{}".format(channel_name.upper()))


def quiete_logs(target="fse"):
    for name in logging.root.manager.loggerDict:
        if "fse" in name:
            logging.getLogger(name=name).setLevel(logging.WARN)


def turn_on_logs(target="fse"):
    for name in logging.root.manager.loggerDict:
        if "fse" in name:
            logging.getLogger(name=name).setLevel(settings.LOG_LEVEL)

