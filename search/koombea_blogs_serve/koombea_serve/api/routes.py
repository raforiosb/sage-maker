import os
import json

from fastapi import FastAPI
from fastapi import HTTPException
from starlette.responses import RedirectResponse
from koombea_serve.ml_model.model import check_health
from koombea_serve.data_module.data import Data
from koombea_serve.config.settings import settings
from koombea_serve.config.logger import logger, logger_error
from koombea_serve.schemas.search import SearchResponseModel, SearchRequestModel

app = FastAPI()
prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").strip("/")


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(prefix + "/docs")


@app.get("/ping")
def ping():
    if check_health():
        return {"status": "ok"}
    else:
        raise HTTPException(status_code=500, detail="ml models not load")


@app.post("/invocations", tags=["Endpoint Search"], response_model=SearchResponseModel)
def search_blog(body: SearchRequestModel):
    logger.info("Search body parameters:\n" + json.dumps(body.dict(), indent=True))
    logger.info("Stage: {}".format(settings.STAGE))

    data = Data(body.s, body.lang)

    try:
        response = data.get_response()
    except Exception as error:
        logger_error(error)
        raise HTTPException(status_code=500, detail=f"error {error}")

    if response:
        try:
            return response.get_response(
                per_page=body.per_page,
                page=body.page,
                content_type=body.content_type,
                term=body.term,
            )
        except Exception as error:
            logger_error(error)
            raise HTTPException(status_code=500, detail=f"error {error}")
    else:
        logger.info("There is no response")
        raise HTTPException(status_code=404, detail="Not result found")
