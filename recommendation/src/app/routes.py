import os, sys, traceback
import json
import time
from redis import Redis
from app.utils import logger
from app.api import my_app as app
from fastapi import Depends, HTTPException
from app.sql_app import schemas, crud, models
from app.sql_app.database import get_session_and_engine
from app.sql_app.database_tunnel import get_tunnel
from app.models import InvocationsRequest, InvocationsResponse
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
from app.dataset.data import Data, check_load_model, get_additional_data

# models.Base.metadata.create_all(bind=engine)
    
def get_db():
    tunnel = get_tunnel()[0]
    tunnel.start()
    db = None
    session = get_session_and_engine(tunnel)[0]["session"]
    if session is not None:
        db = session()
    try:
        yield db
    finally:
        logger.info("closing db and tunnel")
        if db is not None:
            db.close()
        if tunnel is not None:
            tunnel.close()
        
def get_new_db():
    # models.Base.metadata.create_all(bind=engine_prod)
    tunnel = get_tunnel()[1]
    if tunnel is not None: tunnel.start()
    db = None
    session = get_session_and_engine(tunnel_prod=tunnel)[1]["session"]
    if session is not None:
        db = session()
    try:
        yield db
    finally:
        logger.info("closing db and tunnel")
        if db is not None:
            db.close()
        if tunnel is not None:
            tunnel.close()
        
def get_redis_db():
    db = Redis()
    try:
        yield db
    finally:
        db.close()

def logger_error():
    trc = traceback.format_exc()
    logger.error(trc)
    
industries_info, services_info = get_additional_data(get_db)

def is_additional_data(slug):
    if slug in industries_info.keys():
        return True, "industries"
    elif slug in services_info.keys():
        return True, "services"
    else:
        return False, None
    
prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").strip("/")

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(prefix + "/docs")

@app.get("/ping", tags=["Endpoint Health"])
def ping():
    if check_load_model():
        # print(industries_info)
        out = {"status": "ok"}
    else:
        raise HTTPException(status_code=404, detail="artifacts fail to load")
    return out
    

@app.post("/invocations", tags=["Sagemaker Endpoint"], response_model=InvocationsResponse)
def invocations(body: InvocationsRequest, db: Session = Depends(get_db),
               redis_db: Redis = Depends(get_redis_db)):
    # Check if slug is in redis db to get recommendations from cache
    out_recommendations = redis_db.get(body.slug)
    
    if out_recommendations is not None:
        logger.info("Finding in Cache")
        # Check if there is slug to block for certain services
        out_recommendations = json.loads(out_recommendations)
        sw, key_value = is_additional_data(body.slug)
        if body.slug_blocked:
            redis_db.set('blocked_slug_for_'+body.slug, json.dumps(body.slug_blocked))
            if sw and key_value:
                if key_value == "services":
                    out_recommendations = [slug_recommended for slug_recommended in out_recommendations
                                           if not slug_recommended in body.slug_blocked]
        out_recommendations = {"relates":out_recommendations[:body.topn]}
    else:
        # if popular
        if body.slug == 'popular-blogs-api':
            logger.info("finding popular")
            return {"relates": Data.get_recommendations_popular_data(body.topn, "year")}
        if body.slug == 'trending-blogs-api':
            logger.info("finding trending")
            return {"relates": Data.get_recommendations_popular_data(body.topn, "week")}
        # if not get content from database
        sw, key_value = is_additional_data(body.slug)
        if sw and key_value:
            logger.info("Spetial values")
            # Get inferences for services and industries
            try:
                if key_value == "industries":
                    input_request = Data(slug=body.slug, content=industries_info[body.slug],
                                         special_data=True, lang="en", removing_stops=True, lemmatize=True)
                elif key_value == "services":
                    input_request = Data(slug=body.slug, content=services_info[body.slug], 
                                         special_data=True, lang="en", removing_stops=True, lemmatize=True)
                recommendations = input_request.get_recommendations_special_data(body.topn, key_value,
                                                                                 body.slug, slug_blocked=body.slug_blocked)
                out_recommendations = {"relates": recommendations}
            except:
                logger_error()
                return []
        else:
            try:
                db_blog = crud.get_blog_by_slug(db, slug=body.slug)
            except Exception as exception:
                db_blog = None
                logger.info("Error {}".format(exception))
            try:
                if db_blog is None:
                    input_request = Data(slug=body.slug, lang=body.lang, removing_stops=True,
                                         lemmatize=True)
                else:
                    input_request = Data(slug=body.slug, content=db_blog.content, lang=body.lang,
                                        clean_first=True, removing_stops=True, lemmatize=True)
                # logger.info(vars(input_request))
                try:
                    out_recommendations = {"relates":input_request.get_recommendations(body.topn)}
                except:
                    logger_error()
                    return []
            except Exception as error:
                logger_error()
                raise HTTPException(status_code=500, detail=f"Model Error: {error}")
    return out_recommendations