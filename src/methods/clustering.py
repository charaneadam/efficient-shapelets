import numpy as np
from scipy.spatial.distance import cdist
import faiss

from src.data import Windows


class KmeansTransform:
    def __init__(self, window_percentage=30, topk=50, niter=None) -> None:
        self.window_size: int
        self.kmeans = None
        self.ts_length: int
        self.y: np.ndarray
        self.k = topk
        self.window_percentage: int = window_percentage
        self.window_manager: Windows
        self.niter = niter

    def _set_params(self, X, y):
        if self.niter is None:
            self.niter = 20
        self.n_ts, self.ts_length = X.shape
        self.y = y
        self.labels = set(self.y)
        window_size = int(self.ts_length * self.window_percentage / 100)
        self.window_manager = Windows(window_size)
        self.windows = self.window_manager.get_windows(X)
        self.n_centroids = min(500, self.windows.shape[0] // 10)
        self.verbose = True

    def _cluster(self):
        self.kmeans = faiss.Kmeans(
            self.window_manager.size,
            self.n_centroids,
            niter=self.niter,
            verbose=self.verbose,
        )
        self.kmeans.train(self.windows)

        self.dists, self.indices = self.kmeans.index.search(self.windows, 1)
        self.indices = self.indices.reshape(-1)
        clusters = [[] for _ in range(self.n_centroids)]
        for subsequence_index, centroid_index in enumerate(self.indices):
            clusters[centroid_index].append(subsequence_index)
        self.clusters = list(map(np.array, clusters))
        self.shapelets_indices = []

    def _get_window_label(self, window_id):
        ts_id = self.window_manager.get_ts_index_of_window(window_id)
        label = self.y[ts_id]
        return label

    def _select_shapelets(self):
        labels_clusters = {label: [] for label in self.labels}
        for centroid_id, window_ids in enumerate(self.clusters):
            assigned_popularity = 0
            assigned_label = 0
            window_labels = list(map(self._get_window_label, window_ids))
            window_labels = np.array(window_labels)
            for label in self.labels:
                popularity = sum(window_labels == label) / len(window_labels)
                if popularity > assigned_popularity:
                    assigned_popularity = popularity
                    assigned_label = label

            labels_clusters[assigned_label].append((assigned_label, centroid_id))
        for label in self.labels:
            labels_clusters[label] = sorted(labels_clusters[label])
            for i in range(min(self.k, len(labels_clusters[label]))):
                self.shapelets_indices.append(labels_clusters[label][i][1])

    def fit(self, X, y):
        self._set_params(X, y)
        self._cluster()
        self._select_shapelets()

    def transform(self, X):
        windows = self.window_manager.get_windows(X)
        shapelets = self.windows[self.shapelets_indices]
        dists = cdist(windows, shapelets).reshape(
            X.shape[0], -1, len(self.shapelets_indices)
        )
        return dists.min(axis=1)
