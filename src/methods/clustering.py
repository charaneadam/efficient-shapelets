import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import faiss


class KmeansTransform:
    def __init__(self, window_size=30, topk=50) -> None:
        self.window_size: int = window_size
        self.kmeans = None
        self.ts_length: int
        self.y: np.ndarray
        self.k = topk

    def get_windows(self, X):
        """Return Z-Normalized sliding window, it assumes window_size is set.
        It does not set the windows itself, since it sets also test set.
        """
        windows = sliding_window_view(X, window_shape=self.window_size, axis=1)
        windows = windows.reshape(-1, self.window_size).astype("float32")
        return StandardScaler().fit_transform(windows.T).T

    def _set_params(self, X, y):
        self.n_ts, self.ts_length = X.shape
        self.y = y
        self.labels = set(self.y)
        self.windows = self.get_windows(X)
        self.n_centroids = 512
        self.niter = 20
        self.verbose = True

    # def plus_plus(self):
        # centroids = [self.windows[0]]
        # for _ in range(1, self.n_centroids):
            # dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in self.windows])
            # probs = dist_sq/dist_sq.sum()
            # cumulative_probs = probs.cumsum()
            # r = np.random.rand()
            # i = 0
            # for j, p in enumerate(cumulative_probs):
                # if r < p:
                    # i = j
                    # break
            # centroids.append(self.windows[i])
        # return np.array(centroids)

    def _cluster(self):
        # centroids = self.plus_plus()
        self.kmeans = faiss.Kmeans(
            self.window_size, self.n_centroids, niter=self.niter, 
            verbose=self.verbose
        )
        self.kmeans.train(self.windows)
        # self.kmeans.train(self.windows, init_centroids=centroids)

        self.dists, self.indices = self.kmeans.index.search(self.windows, 1)
        self.indices = self.indices.reshape(-1)
        clusters = [[] for _ in range(self.n_centroids)]
        for subsequence_index, centroid_index in enumerate(self.indices):
            clusters[centroid_index].append(subsequence_index)
        self.clusters = list(map(np.array, clusters))
        self.shapelets_indices = []

    def _get_window_label(self, window_id):
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        ts_id = window_id // number_of_windows_per_ts
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
        windows = self.get_windows(X)
        shapelets = self.windows[self.shapelets_indices]
        dists = cdist(windows, shapelets).reshape(
            X.shape[0], -1, len(self.shapelets_indices)
        )
        return dists.min(axis=1)
