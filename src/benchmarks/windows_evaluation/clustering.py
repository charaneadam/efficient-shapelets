import numpy as np
from time import perf_counter
from numba import njit, prange
import faiss

from src.benchmarks.windows_evaluation.db import save
from src.benchmarks.windows_evaluation.utils import distance_numba
from src.storage.data import Data, Windows


@njit(cache=True, fastmath=True)
def _eval_clustering(X, y, windows, windows_labels):
    n_windows = windows.shape[0]
    n_ts = X.shape[0]
    res = np.zeros(n_windows)
    for window_id in prange(n_windows):
        window = windows[window_id]
        a, b = 0.0, 0.0
        a_tot = 0
        b_tot = 0
        for ts_id in prange(n_ts):
            dist = distance_numba(X[ts_id], window)
            if y[ts_id] == windows_labels[window_id]:
                a += dist
                a_tot += 1
            else:
                b += dist
                b_tot += 1
        a /= a_tot
        b /= b_tot
        if b > a:
            mx = b
        else:
            mx = a
        res[window_id] = (b - a) / mx
    return res


@njit(cache=True, fastmath=True)
def assign_labels_to_clusters(n_centroids, labels, windows_indices, y, windows_per_ts):
    count = np.zeros((n_centroids, len(labels)))
    labels_remap = dict(zip(labels, range(len(labels))))
    n_windows = windows_indices.shape[0]
    for window_id in prange(n_windows):
        centroid_id = windows_indices[window_id]
        count[centroid_id][labels_remap[y[(window_id // windows_per_ts)]]]
    return labels[count.argmax(axis=1)]


def cluster(data: Data, window_manager: Windows):
    windows = window_manager.get_windows(data.X_train)
    start = perf_counter()
    n_centroids = min(500, windows.shape[0] // 20)
    kmeans = faiss.Kmeans(window_manager.size, n_centroids, niter=3)
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
    save(
        data.dataset_name,
        "Clustering",
        window_manager.size,
        window_manager.skip,
        cluster_time + eval_time,
        results,
        centroids_labels
    )


if __name__ == "__main__":
    from src.storage.database import db
    from .db import init_windows_tables

    init_windows_tables(db)

    dataset_name = "CBF"
    data = Data(dataset_name)
    windows_size = 40
    window_skip = int(0.1 * windows_size)
    window_manager = Windows(windows_size, window_skip)
    cluster(data, window_manager)
