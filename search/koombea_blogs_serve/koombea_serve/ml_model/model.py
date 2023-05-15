from koombea_serve.data_module.data_cleaning_utils import (
    loaded_from_path_clean_df,
    loaded_from_redis_clean_df,
)
from koombea_serve.ml_model.search_response import SearchResponse

import os
import json
from typing import List
import faiss
import pickle
import fse

import pandas as pd
import numpy as np

from redis import Redis
from koombea_serve.config.settings import settings
from koombea_serve.config.logger import logger
from koombea_serve.db.deps import get_redis_connection


class Model:
    model_artifacts = {
        "bv_model_en": None,
        "bv_model_es": None,
        "bv_matrix_en": None,
        "bv_matrix_es": None,
    }

    db_artifacts = {
        "blogs_df": None,
        "idx2ids_mapping_es": None,
        "idx2ids_mapping_en": None,
    }

    @classmethod
    def get_model_artifact_by_lang(cls, lang: str):
        if cls.model_artifacts[f"bv_model_{lang}"] is None:
            cls.model_artifacts[f"bv_model_{lang}"] = fse.models.SIF.load(
                os.path.join(
                    settings.SM_MODEL_DIR,
                    "bv_model_{}_{}.bv".format(lang, settings.MYSQL_DBNAME),
                )
            )
        for redis_conn in get_redis_connection():
            if redis_conn.get(f"bv_matrix_{lang}") is not None:
                logger.info("loading from redis cache")
                cls.model_artifacts[f"bv_matrix_{lang}"] = pickle.loads(
                    redis_conn.get(f"bv_matrix_{lang}")
                )
            else:
                cls.model_artifacts[f"bv_matrix_{lang}"] = cls.model_artifacts[
                    f"bv_model_{lang}"
                ].sv.vectors

        # adding indexes to faiss
        faiss.normalize_L2(cls.model_artifacts[f"bv_matrix_{lang}"])
        cls.model_artifacts[f"index_model_{lang}"] = faiss.IndexFlatIP(
            cls.model_artifacts[f"bv_matrix_{lang}"].shape[1]
        )
        cls.model_artifacts[f"index_model_{lang}"].add(
            cls.model_artifacts[f"bv_matrix_{lang}"]
        )

    @classmethod
    def get_model_artifacts(cls):
        cls.get_model_artifact_by_lang(lang="en")
        cls.get_model_artifact_by_lang(lang="es")
        return cls.model_artifacts

    @classmethod
    def get_db_artifacts(cls):
        for redis_conn in get_redis_connection():
            if redis_conn.get("blogs_df") is not None:
                # there is data in redis db
                logger.info("loading from redis cache")
                for item_name in cls.db_artifacts.keys():
                    if "blogs" in item_name:
                        blogs_df = pd.DataFrame.from_dict(
                            json.loads(redis_conn.get(item_name))
                        )
                        cls.db_artifacts[item_name] = loaded_from_redis_clean_df(blogs_df)
                    else:
                        cls.db_artifacts[item_name] = json.loads(redis_conn.get(item_name))

            else:
                for item_name, item_object in cls.db_artifacts.items():
                    if item_object is None:
                        if "blogs" in item_name:
                            blogs_df = pd.read_csv(
                                os.path.join(
                                    settings.SM_MODEL_DIR,
                                    "{}_{}.csv".format(item_name, settings.MYSQL_DBNAME),
                                )
                            )
                            # clean df
                            cls.db_artifacts[item_name] = loaded_from_path_clean_df(
                                blogs_df
                            )
                        else:
                            with open(
                                os.path.join(
                                    settings.SM_MODEL_DIR,
                                    "{}_{}.json".format(item_name, settings.MYSQL_DBNAME),
                                ),
                                "r",
                            ) as json_file:
                                cls.db_artifacts[item_name] = json.load(json_file)
            return cls.db_artifacts

    @classmethod
    def get_artifacts(cls):
        return {**cls.get_model_artifacts(), **cls.get_db_artifacts()}

    @classmethod
    def infer_vector(cls, data: List[str], lang: str) -> np.ndarray:
        artifacts = cls.get_artifacts()
        bv_model: fse.models.SIF = (
            artifacts["bv_model_en"] if lang == "en" else artifacts["bv_model_es"]
        )
        query_vector = bv_model.infer([(data, 0)])
        return query_vector.reshape(-1)

    @classmethod
    def get_results(cls, query: List[str], lang: str):
        artifacts = cls.get_artifacts()

        bv_model: fse.models.SIF = (
            artifacts["bv_model_en"] if lang == "en" else artifacts["bv_model_es"]
        )
        blogs_df: pd.DataFrame = artifacts["blogs_df"]
        blogs_df = blogs_df[blogs_df["lang"] == lang]
        index_model = (
            artifacts["index_model_en"] if lang == "en" else artifacts["index_model_es"]
        )

        logger.info("query: {}".format(query))

        query_vector = bv_model.infer([(query, 0)])
        faiss.normalize_L2(query_vector)

        sw_results = not ((np.zeros_like(query_vector) == query_vector).sum() == 300)
        if sw_results:
            D, I = index_model.search(query_vector, k=blogs_df.shape[0])
            D, I = D.squeeze(), I.squeeze()
            response = SearchResponse(I, 1 - D, blogs_df, lang=lang)
            return response
        else:
            logger.info("There is no results for the current query")
            return None


def check_health():
    logger.info("checking health")
    health = [artifact is not None for artifact in Model.get_artifacts().values()]
    return all(health)
