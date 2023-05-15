from koombea_model.vector_model.blogs_model import Top
from pandas.core.indexes.numeric import Int64Index
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Union
import functools
import random
import numpy as np


def train_tfidf(data: List[Tuple[int, List[str]]]):
    _, tokens_blogs = zip(*data)
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split())
    tfidf_matrix = tfidf.fit_transform(
        (" ".join(tokens_blog) for tokens_blog in tokens_blogs)
    )
    return tfidf_matrix, tfidf.get_feature_names()


def train_mapping_reduction_cluster(blogs_vectors: np.ndarray, n_clusters: int):
    tsne = TSNE(n_components=2).fit_transform(blogs_vectors)
    c = KMeans(n_clusters=n_clusters).fit_predict(blogs_vectors)
    return tsne, c


def get_random_stratify(clusters: np.ndarray, total_points: int, n_clusters: int):
    if total_points > 15:
        partitions = (
            total_points // n_clusters
            if not total_points % n_clusters
            else total_points // n_clusters + 1
        )
    else:
        partitions = len(clusters)
    # get groups
    groups = {i: [] for i in np.unique(clusters)}
    for idx, val in enumerate(clusters):
        groups[val].append(idx)
    # return indexes selected randomly in each partition
    return functools.reduce(
        lambda x, y: x + random.choices(y[1], k=partitions), groups.items(), []
    )


def compute_accuracy(
    targets: Union[List[Top], Top], cache_results: List[Tuple[int, Int64Index]]
):
    if isinstance(targets, Top):
        targets = [targets]
    accuracies_top = [
        [(target.name, id_target in ids_result[:target]) for target in targets]
        for id_target, ids_result in cache_results
    ]
    accuracies_top = {
        acc_list[0][0]: functools.reduce(lambda x, y: x + int(y[1]), acc_list, 0)
        / len(acc_list)
        for acc_list in zip(*accuracies_top)
    }
    return accuracies_top


def calculate_distance(
    cache_results: List[Tuple[int, Int64Index]],
    targets: Union[List[Top], Top],
    top_distance_bad_good: int,
    n_top: int,
):
    distances_per_target = [
        [
            (
                target.name,
                id_target,
                ids_result[:n_top].tolist(),
                target - ids_result.get_indexer_for([id_target]).item(),
            )
            for target in targets
        ]
        for id_target, ids_result in cache_results
    ]

    def calculate_bads_goods(acc_list, top_distance_bad_good: int):
        acc_list = sorted(acc_list, key=lambda x: x[-1])
        return [
            (acc_list[i][1:], acc_list[-i - 1][1:])
            for i in range(top_distance_bad_good)
        ]

    distances_top = {
        acc_list[0][0]: calculate_bads_goods(acc_list, top_distance_bad_good)
        for acc_list in zip(*distances_per_target)
    }
    return distances_top
