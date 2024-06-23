import numpy as np
from time import perf_counter
from numba import njit, prange, objmode
import faiss

from .db import save
from .utils import distance_numba, silhouette, fstat, info_gain
from src.storage.data import Data, Windows


@njit(cache=True, fastmath=True)
def _eval_clustering(X, y, windows, windows_labels):
    n_windows = windows.shape[0]
    n_ts = X.shape[0]
    res = np.zeros((n_windows, 6))  # 6: 3 for sil,infogain,fstat, and 3 for time
    for window_id in prange(n_windows):
        window = windows[window_id]
        window_label = windows_labels[window_id]
        dists_to_ts = np.zeros(n_ts)
        for ts_id in prange(n_ts):
            dists_to_ts[ts_id] = distance_numba(X[ts_id], window)

        with objmode(start="f8"):
            start = perf_counter()
        silhouette_score = silhouette(dists_to_ts, window_label, y)
        with objmode(end="f8"):
            end = perf_counter()
        silhouette_time = end - start
        res[window_id][0] = silhouette_score
        res[window_id][1] = silhouette_time

        with objmode(start="f8"):
            start = perf_counter()
        fstat_score = fstat(dists_to_ts, window_label, y)
        with objmode(end="f8"):
            end = perf_counter()
        fstat_time = end - start
        res[window_id][2] = fstat_score
        res[window_id][3] = fstat_time

        with objmode(start="f8"):
            start = perf_counter()
        infgain_score = info_gain(dists_to_ts, window_label, y)
        with objmode(end="f8"):
            end = perf_counter()
        infogain_time = end - start
        res[window_id][4] = infgain_score
        res[window_id][5] = infogain_time
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
    _, indices = kmeans.index.search(windows, 1)
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
        centroids_labels,
    )


if __name__ == "__main__":
    dataset_name = "CBF"
    data = Data(dataset_name)
    windows_size = 40
    window_skip = int(0.1 * windows_size)
    window_manager = Windows(windows_size, window_skip)
    cluster(data, window_manager)
