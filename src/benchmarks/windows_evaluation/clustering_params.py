from numba import njit
import numpy as np
from time import perf_counter
import faiss
from src.benchmarks.get_experiment import get_approach_id, get_datasets
from src.storage.data import Data, Windows
from .db import (
    init_clustering_tables,
    save_clustering_parameters,
)
from .clustering import assign_labels_to_clusters, _eval_clustering


def get_missing_centroids(dataset, approach_id):
    return {}


@njit(fastmath=True, cache=True, parallel=True)
def centroids_info(y, indices, distances, centroids_labels, n_windows_per_ts):
    n_centroids = len(centroids_labels)
    info = np.zeros((n_centroids, 6))
    for centroid_id in range(n_centroids):
        windows_indices = np.where(indices == centroid_id)
        windows_dists = distances[windows_indices]
        windows_ts_indices = windows_indices // n_windows_per_ts
        windows_labels = y[windows_ts_indices]

        population_size = windows_indices.shape[0]
        same_windows_index = windows_labels == centroids_labels[centroid_id]
        popularity = np.sum(same_windows_index)
        popularity /= population_size
        distinct_ts = len(set(windows_ts_indices[same_windows_index]))
        avg_dist_same = np.mean(windows_dists[same_windows_index])
        avg_dist_diff = np.nan
        if np.any(~same_windows_index):
            avg_dist_diff = np.mean(windows_dists[~same_windows_index])
        info[centroid_id][0] = population_size
        info[centroid_id][1] = popularity
        info[centroid_id][2] = distinct_ts
        info[centroid_id][3] = avg_dist_same
        info[centroid_id][4] = avg_dist_diff
        info[centroid_id][5] = centroids_labels[centroid_id]
    return info


def cluster(data, window_manager, n_centroids, dataset, n_iterations=10):
    windows = window_manager.get_windows(data.X_train)
    start = perf_counter()
    kmeans = faiss.Kmeans(window_manager.size, n_centroids, niter=n_iterations)
    kmeans.train(windows)
    dists, indices = kmeans.index.search(windows, 1)
    indices = indices.reshape(-1)
    centroids_labels = assign_labels_to_clusters(
        n_centroids,
        np.array(list(set(data.y_test))),
        indices,
        data.y_train,
        window_manager.windows_per_ts,
    )
    end = perf_counter()
    cluster_time = end - start

    start = perf_counter()
    results = _eval_clustering(
        data.X_train, data.y_train, kmeans.centroids, centroids_labels
    )
    end = perf_counter()
    eval_time = end - start

    start = perf_counter()
    info_results = centroids_info(
        y=data.y_train,
        indices=indices,
        distances=dists,
        centroids_labels=centroids_labels,
        n_windows_per_ts=window_manager.n_windows_per_ts,
    )
    end = perf_counter()
    centroids_info_time = end - start

    return cluster_time, results, eval_time, info_results, centroids_info_time


def run():
    datasets = get_datasets()
    approach_id = get_approach_id("Clustering")
    for dataset in datasets:
        missing_centroids = get_missing_centroids(dataset, approach_id)
        if len(missing_centroids) == 0:
            continue
        data = Data(dataset.name)
        for window_size in missing_centroids.keys():
            window_skip = int(0.1 * window_size)
            window_manager = Windows(window_size, window_skip)
            for n_centroids in missing_centroids[window_size]:
                try:
                    (
                        cluster_time,
                        results,
                        eval_time,
                        info_results,
                        centroids_info_time,
                    ) = cluster(data, window_manager, n_centroids, dataset)
                    save_clustering_parameters(
                        dataset=dataset,
                        window_size=window_size,
                        skip_size=window_skip,
                        n_iterations=None,
                        clustering_time=cluster_time,
                        centroids_evaluation=results,
                        evaluation_time=eval_time,
                        centroids_info=info_results,
                        info_time=centroids_info_time,
                    )
                except:
                    print("error", end=", ")


if __name__ == "__main__":
    from src.storage.database import db
    init_clustering_tables(db)
