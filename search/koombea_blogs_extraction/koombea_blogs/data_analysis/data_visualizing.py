from typing import List, Tuple
import nltk
import numpy as np
from functools import reduce
from ..data_analysis import visualize_utils
from ..config.deps import get_wandb_run_visualizing_analysis
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.decomposition as deco
import wandb


class DataVisualizing:
    def __init__(
        self,
        english_data: List[List[str]],
        spanish_data: List[List[str]],
        num_topic_en: int = 20,
        num_topic_es: int = 2,
        settings = None
    ) -> None:
        self.en_data = english_data
        self.es_data = spanish_data
        self.settings = settings
        self.wandb_run = get_wandb_run_visualizing_analysis(self.settings)
        # get tfidf model
        self.tfidf_en = TfidfVectorizer().fit(
            [" ".join(tokens) for tokens in self.en_data]
        )
        self.tfidf_es = TfidfVectorizer().fit(
            [" ".join(tokens) for tokens in self.es_data]
        )
        self.matrix_en = self.tfidf_en.transform(
            [" ".join(tokens) for tokens in self.en_data]
        )
        self.matrix_es = self.tfidf_es.transform(
            [" ".join(tokens) for tokens in self.es_data]
        )
        # get nmf decomposition model
        self.nmf_en = deco.NMF(n_components=num_topic_en, init="nndsvd", max_iter=1000)
        self.nmf_en.fit(self.matrix_en)
        self.nmf_es = deco.NMF(n_components=num_topic_es, init="nndsvd", max_iter=1000)
        self.nmf_es.fit(self.matrix_es)

    def make_frequency_analysis(
        self,
        title: str,
        sup_titles: Tuple[str, str],
        lang: str = "en",
        n_top: int = 20,
        **kwargs,
    ):
        data = self.en_data if lang == "en" else self.es_data
        data = reduce(lambda l1, l2: l1 + l2, data)
        words_fdist = nltk.probability.FreqDist(data)
        fig = visualize_utils.plot_frequency_words_info(
            words_fdist,
            n_top=n_top,
            title=title,
            sup_titles=sup_titles,
            **kwargs,
        )
        key = f"frequency_chart_{lang}"
        self.wandb_run.log({key: wandb.Plotly(fig)})

    def make_tfidf_mean_analysis(
        self,
        title: str,
        sup_titles: Tuple[str, str],
        lang: str = "en",
        n_top: int = 30,
        **kwargs,
    ):
        vocab = (
            self.tfidf_en.get_feature_names()
            if lang == "en"
            else self.tfidf_es.get_feature_names()
        )
        mean_weights = np.array(
            (self.matrix_en if lang == "en" else self.matrix_es).mean(axis=0)
        ).reshape(-1)
        fig = visualize_utils.plot_tfidf_mean_words_info(
            vocab=vocab,
            mean_weights=mean_weights,
            n_top=n_top,
            title=title,
            sup_titles=sup_titles,
            **kwargs,
        )
        self.wandb_run.log({f"tfidf_chart_mean_{lang}": wandb.Plotly(fig)})

    def make_tfidf_by_topics_analysis(
        self,
        title: str,
        lang: str = "en",
        n_top: int = 3,
        **kwargs,
    ):
        nmf = self.nmf_en if lang == "en" else self.nmf_es
        vocab = (
            self.tfidf_en.get_feature_names()
            if lang == "en"
            else self.tfidf_es.get_feature_names()
        )
        fig = visualize_utils.plot_tfidf_topic_words_info(
            nmf, vocab=vocab, title=title, n_top=n_top, **kwargs
        )
        self.wandb_run.log({f"tfidf_chart_topics_{lang}": wandb.Plotly(fig)})

    def make_tfidf_wordcloud_by_topics_analysis(
        self,
        title: str,
        lang: str = "en",
        n_top: int = 20,
        **kwargs,
    ):
        nmf = self.nmf_en if lang == "en" else self.nmf_es
        vocab = (
            self.tfidf_en.get_feature_names()
            if lang == "en"
            else self.tfidf_es.get_feature_names()
        )
        image = visualize_utils.plot_wordcloud_tfidif_info(
            nmf, vocab, title=title, n_top=n_top, **kwargs
        )
        self.wandb_run.log({f"wordcloud_tfidf_images_topics_{lang}": image})
