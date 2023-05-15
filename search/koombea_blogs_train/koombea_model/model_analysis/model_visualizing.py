import pandas as pd
import numpy as np

from typing import List, Tuple, Union

import wandb
from koombea_model.data import Data
from koombea_model.config.logger import logger, get_wandb_run_training
from koombea_model.vector_model.blogs_model import BlogsModel, Top
from koombea_model.model_analysis.analize_utils import (
    train_tfidf,
    train_mapping_reduction_cluster,
    get_random_stratify,
    calculate_distance,
    compute_accuracy,
)
from koombea_model.model_analysis.visualize_utils import (
    plot_accuracies_information_df,
    plot_blogs_embedding_information,
)


class ModelVisualizeAnalysis:
    def __init__(self, bv_model_en: BlogsModel, bv_model_es: BlogsModel, data: Data):
        self.bv_model_en = bv_model_en
        self.bv_model_es = bv_model_es
        self.data = data
        logger.info("Training tfidf for model analysis and visualizing")
        # train tfidf for english
        self.tfidf_matrix_en, self.vocab_en = train_tfidf(self.data.en_data)
        # train tfidf for spanish
        self.tfidf_matrix_es, self.vocab_es = train_tfidf(self.data.es_data)
        # wandb logger
        self.run = get_wandb_run_training()

    def model_summary(
        self,
        targets: Union[List[Top], Top],
        lang: str = "en",
        top_distance_bad_good: int = 10,
        n_top: int = 5,
        top_words: int = 3,
        n_clusters: int = 20,
        total_points: int = 80,
    ):
        logger.debug("generating model summary for {} model".format(lang))
        logger.info("getting model summary tables")
        summary_df, accuracies_df = self._describe_model_tables(
            targets, lang, top_distance_bad_good, n_top=n_top
        )
        logger.info("prepare embeddings configuration for visualization charts")
        data_plot_config = self._prepare_embeddings_configuration(
            lang, total_points, top_n=top_words, n_clusters=n_clusters
        )
        logger.info("generate accuracy chart")
        accuracy_fig = plot_accuracies_information_df(
            accuracies_df, title="Accuracy results per top for {} blogs".format(lang)
        )
        logger.info("generate blogs embedding chart")
        embeddings_fig = plot_blogs_embedding_information(
            data_plot_config,
            marker_size=13,
            title="blogs vector embedding for {} blogs".format(lang),
        )
        # log tables and figures
        self.run.log(
            {
                f"summary_df_{lang}": wandb.Table(dataframe=summary_df),
                f"accuracies_df_{lang}": wandb.Table(dataframe=accuracies_df),
                f"embeddings_fig_{lang}": wandb.Plotly(embeddings_fig),
                f"accuracy_fig_{lang}": wandb.Plotly(accuracy_fig),
            }
        )

    def _prepare_embeddings_configuration(
        self,
        lang: str = "en",
        total_points: int = 80,
        top_n: int = 3,
        n_clusters: int = 20,
    ):
        # Get arguments acording to language
        bv_model = (
            self.bv_model_en.b2v_model if lang == "en" else self.bv_model_es.b2v_model
        )
        vocab = self.vocab_en if lang == "en" else self.vocab_es
        tfidf_matrix = self.tfidf_matrix_en if lang == "en" else self.tfidf_matrix_es
        # validate n_clusters
        if n_clusters > tfidf_matrix.shape[0]:
            n_clusters = tfidf_matrix.shape[0] // 2
        # fit tsne and clusters
        tsne, c = train_mapping_reduction_cluster(
            bv_model.sv.vectors, n_clusters=n_clusters
        )
        # get random selected blogs
        random_blogs_idx = get_random_stratify(c, total_points, n_clusters)
        # get important words
        word_idx_matrix = tfidf_matrix.toarray()[random_blogs_idx].argsort(axis=1)[
            :, -top_n:
        ]
        words = [list(np.array(vocab)[i]) for i in word_idx_matrix]
        return {
            "words": words,
            "tsne": tsne,
            "clusters": c,
            "blogs_idx_selected": random_blogs_idx,
        }

    def _describe_model_tables(
        self,
        targets: Union[List[Top], Top],
        lang: str = "en",
        top_distance_bad_good: int = 10,
        n_top: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get arguments acording to language
        blogs_df = self.data.blogs_df[self.data.blogs_df["lang"] == lang]
        cache_results = (
            self.bv_model_en.cache_predictions
            if lang == "en"
            else self.bv_model_es.cache_predictions
        )
        # get accuracy for each top
        accuracies_top = compute_accuracy(targets, cache_results)
        # get top blogs with highest distance bad and good this is only necessary for only one top
        distances_top = calculate_distance(
            cache_results,
            targets=[max(targets)],
            top_distance_bad_good=top_distance_bad_good,
            n_top=n_top,
        )
        # calculate distance and make a summary table
        distances_data = [
            (
                result_type + "_result",
                blog_id,
                blogs_df[blogs_df["ID"] == blog_id].post_title.item(),
                metric_val,
                *[
                    blogs_df[blogs_df["ID"] == best_id].post_title.item()
                    for best_id in best_results
                ],
            )
            for _, values_type in distances_top.items()
            for result_type, values in zip(["bad", "good"], zip(*values_type))
            for blog_id, best_results, metric_val in values
        ]
        distances_columns = [
            "top_type",
            "expected_id",
            "expected_blog_slug",
            "result_metric",
            *[f"top_{i + 1}" for i in range(n_top)],
        ]
        return (
            pd.DataFrame(data=distances_data, columns=distances_columns),
            pd.DataFrame(
                data=[val for val in accuracies_top.values()],
                index=[key for key in accuracies_top.keys()],
                columns=["accuracy_results"],
            ),
        )
