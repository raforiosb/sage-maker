from typing import List, Tuple
from gensim.models import Word2Vec
from pathlib import Path

from pandas.core.indexes.numeric import Int64Index
import wandb
from ..config.logger import logger, get_wandb_run_training
from ..config.settings import settings
from koombea_model.general_utils import pretty_string
from tqdm.auto import tqdm
from ..data_cleaning_utils import patterns_replacement, process_data
from scipy.spatial import distance
import pandas as pd
from enum import Enum
import fse


class Top(int, Enum):
    top3 = 3
    top5 = 5
    top10 = 10
    top15 = 15


class BlogsModel:
    def __init__(self, name: str, lang: str, output_path: Path) -> None:
        self.name = name + "_" + lang + "_" + settings.MYSQL_DBNAME + ".bv"
        self.lang = lang
        self.output_path = output_path
        self.wandb_logger = get_wandb_run_training()

    def _train_word_vectors(
        self, data: List[Tuple[int, List[str]]], **parameters
    ) -> Word2Vec:
        """Train a word vectors models given tokenized and preprocessed blogs dataset

        Args:
            data (List[Tuple[int, List[str]]]): List of tuples with (blogs id, List ot tokens)

        Returns:
            Word2Vec: word vectors model
        """
        _, data = zip(*data)
        model = Word2Vec(**parameters)
        model.build_vocab(data)
        model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def _train_blog_vectors(
        self,
        data: List[Tuple[int, List[str]]],
        word_vector_model: Word2Vec,
        **parameters,
    ) -> fse.models.SIF:
        """Train blog vectors model

        Args:
            data (List[Tuple[int, List[str]]]): List of tuples with (blogs id, List ot tokens)
            word_vector_model (Word2Vec): word vectors trained model

        Returns:
            fse.models.SIF: Smooth inverse frequency trained embedding model
        """
        _, data = zip(*data)
        fse_data = fse.IndexedList(list(data))
        fse_model = fse.models.SIF(word_vector_model, **parameters)
        fse_model.train(fse_data)
        return fse_model

    def train(self, data: List[Tuple[int, List[str]]], components=0, **parameters):
        """Train blogs model, first we train a word vectors model using w2v algorithm, then
        train a sif model.

        Args:
            data (List[Tuple[int, List[str]]]): List of data with tuples (blogs_id, list of tokens)
            components (int, optional): number of components to remove (singular vectors decomposition). Defaults to 0.
        """
        logger.info(
            f"Training blogs model for {len(data)} blogs of language: {self.lang} using the following hyperparameters:\n"
            f"{pretty_string(parameters)}"
        )
        w2v_model = self._train_word_vectors(data, **parameters)
        self.b2v_model = self._train_blog_vectors(
            data,
            w2v_model,
            components=components,
            workers=parameters["workers"],
            lang_freq=self.lang,
        )

    def save(self) -> None:
        """Save model on disk"""
        # need to save the model, for serving and validation
        logger.info(f"Saving model to {self.output_path/ self.name}")
        output_path = self.output_path / self.name
        self.b2v_model.save(str(output_path))

    def cache_results(
        self, blogs_df: pd.DataFrame, ids_mapping: List[int], max_top: int = 100
    ):
        logger.info("getting cache results prediction...")
        blogs_df_temp = blogs_df[blogs_df["lang"] == self.lang]
        self.cache_predictions: List[Tuple[int, Int64Index]] = [
            (
                row.ID,
                self.show_query_similarity_table(
                    blogs_df_temp, row.post_title, ids_mapping, top_n=max_top
                ).index,
            )
            for row in tqdm(
                blogs_df_temp.itertuples(),
                total=blogs_df_temp.shape[0],
                desc=f"Extracting {max_top} predictions",
            )
        ]

    def show_word_similarity_table(self, *words: str, top_n: int = 10) -> pd.DataFrame:
        index = list(words)
        get_topn_words = lambda word, top_n: list(
            zip(*self.b2v_model.wv.most_similar(word, topn=top_n))
        )[0]
        data = [get_topn_words(word, top_n) for word in words]
        return pd.DataFrame(
            index=index,
            data=data,
            columns=["top_{}".format(i + 1) for i in range(top_n)],
        )

    def show_query_similarity_table(
        self,
        blogs_df: pd.DataFrame,
        query: str,
        ids_mapping: List[int],
        top_n: int = 10,
    ) -> pd.DataFrame:
        query = process_data(
            query,
            self.lang,
            lemmatize=True,
            remove_stops=True,
            patterns_replacement=patterns_replacement,
        )
        query_vector = self.b2v_model.infer([(query, 0)])
        similarity_results = distance.cdist(
            query_vector, self.b2v_model.sv.vectors, metric="cosine"
        ).squeeze()
        idx_sorted = similarity_results.argsort()
        results = [
            [
                blogs_df[blogs_df["ID"] == ids_mapping[idx]]["post_title"].item(),
                1 - value,
            ]
            for idx, value in zip(idx_sorted, similarity_results[idx_sorted])
        ][:top_n]
        index = [
            ids_mapping[idx]
            for idx, _ in zip(idx_sorted, similarity_results[idx_sorted])
        ][:top_n]
        return pd.DataFrame(
            index=index, data=results, columns=["post_title", "similarity_value"]
        )

    def show_blogs_similarity_table(
        self,
        blogs_df: pd.DataFrame,
        blog_id: int,
        ids_mapping: List[int],
        top_n: int = 10,
    ) -> pd.DataFrame:
        ids2idx = {ids: idx for idx, ids in enumerate(ids_mapping)}
        similarity_results = self.b2v_model.sv.most_similar(
            ids2idx[blog_id], topn=top_n
        )
        results = [
            [blogs_df[blogs_df["ID"] == ids_mapping[idx]]["post_title"].item(), value]
            for idx, value in similarity_results
        ]
        return pd.DataFrame(
            index=[ids_mapping[idx] for idx, _ in similarity_results[:top_n]],
            data=results,
            columns=["post_title", "similarity_value"],
        )

    def build_metadata(self) -> None:
        artifact = wandb.Artifact(
            name=self.name,
            type="model",
            description="model fse artifact data file",
            metadata={"db_stage": settings.STAGE, "db_name": settings.MYSQL_DBNAME},
        )
        output_path = self.output_path / self.name
        artifact.add_file(str(output_path))
        self.wandb_logger.log_artifact(artifact)
