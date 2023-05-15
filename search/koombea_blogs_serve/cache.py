#! /usr/bin/env python
from typing import List, Tuple

import json
import redis
import numpy as np
import pandas as pd

import pickle

from koombea_serve.config.logger import logger
from koombea_serve.config.settings import settings
from koombea_serve.db.deps import get_connection, get_redis_connection
from koombea_serve.data_module.data_extraction import DataExtraction
from koombea_serve.data_module.data_processing import DataProcessing
from koombea_serve.data_module.data_cleaning_utils import convert_date
from koombea_serve.ml_model.model import Model


def cache_matrix(data: List[Tuple[int, str]], lang: str, redis_conn):
    idx2ids_mapping, tokens_list = zip(*data)
    redis_conn.set(f"idx2ids_mapping_{lang}", json.dumps(idx2ids_mapping))

    bv_matrix = np.zeros(shape=(len(tokens_list), 300), dtype=np.float32)
    for index, tokens in enumerate(tokens_list):
        bv_matrix[index] = Model.infer_vector(tokens, lang)
    redis_conn.set(f"bv_matrix_{lang}", pickle.dumps(bv_matrix))


def cache_dataset(blogs: pd.DataFrame, redis_conn):
    blogs["post_date"] = blogs["post_date"].apply(convert_date)
    blogs["post_modified"] = blogs["post_modified"].apply(convert_date)

    blogs = blogs.to_dict("records")
    redis_conn.set("blogs_df", json.dumps(blogs))


if __name__ == "__main__":
    logger.info("Extract Data")
    logger.info("Initialize connection to db: {}".format(settings.MYSQL_DBNAME))
    # extract data
    for conn, redis_conn in zip(get_connection(), get_redis_connection()):
        extraction = DataExtraction(conn)
        extraction.extract()
        # process data
        processing = DataProcessing(extraction.blogs)
        processing.preprocess_data()
        # cache matrix and mapping idx
        cache_matrix(processing.english_data, lang="en", redis_conn=redis_conn)
        cache_matrix(processing.spanish_data, lang="es", redis_conn=redis_conn)
        # cache dataset
        cache_dataset(extraction.blogs, redis_conn=redis_conn)
