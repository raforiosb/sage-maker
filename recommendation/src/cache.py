#! /usr/bin/env python
from redis import Redis
from app.utils import logger
from app.routes import get_redis_db, get_db, get_new_db
# from app.routes import get_db
from app.dataset.data import Dataset, AnalyticDataset


# def main(redis_db, db):
#     # dataset_en, dataset_es = Dataset.create_dataset(redis_db, db)
#     dataset_en.predict('en')
#     dataset_es.predict('es')

def main(redis_db, db, new_db):
    dataset_en, dataset_es = Dataset.create_dataset_(redis_db, db)
    dataset_en.predict('en')
    dataset_es.predict('es')
    # new_db = list(get_new_db())[0]
    
    if new_db is not None:
        ga_dataset = AnalyticDataset.create_analytics_dataset(redis_db,
                                                         type_analysis="year",
                                                         db = new_db)
        ga_dataset.cache_populars()
        
        ga_dataset = AnalyticDataset.create_analytics_dataset(redis_db,
                                                         type_analysis="week",
                                                         db = new_db)
    else:
        ga_dataset = AnalyticDataset.create_analytics_dataset(redis_db,
                                                         type_analysis="year",
                                                         db = db)
        ga_dataset.cache_populars()
        
        ga_dataset = AnalyticDataset.create_analytics_dataset(redis_db,
                                                         type_analysis="week",
                                                         db = db)
    ga_dataset.cache_trending()
    
if __name__ == "__main__":
    logger.info("Begin Cache All Prediction")
    # redis_db = list(get_redis_db())[0]
    # db  = list(get_db())[0]
    # main(redis_db, db)
    for redis_db, db, new_db in zip(get_redis_db(), get_db(), get_new_db()):
        main(redis_db, db, new_db)
    logger.info("Caching process ends")
    
    