from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import faiss


class NearestNeighborTransform:
    def __init__(self, verbose=0, threshold=0.8, n_neighbors=10, window_size=30):
        self.window_size: int | None = None
        self.index = None
        self.X = None
        self.y = None
        self.verbose = verbose
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.window_size = window_size

    def _get_windows(self, X):
        windows = sliding_window_view(X, window_shape=self.window_size, axis=1)
        windows = windows.reshape(-1, self.window_size).astype("float32")
        windows = StandardScaler().fit_transform(windows.T).T
        return windows

    def _set_params(self, X, y):
        self.X, self.y = X, y
        self.n_ts, self.ts_length = X.shape

        if self.verbose:
            print(f"Window length: {self.window_size}")

        self.windows = self._get_windows(X)
        self.n_windows = self.windows.shape[0]

        # self.index = faiss.IndexFlatL2(self.window_size)
        self.index = faiss.IndexHNSWFlat(self.window_size, self.n_neighbors)
        self.index.add(self.windows)

        self.window_size = min(self.window_size, int(0.4 * self.X.shape[1]))
        if self.verbose:
            print(f"# neighbors: {self.n_neighbors}")

    def fit(self, X, y):
        self._set_params(X, y)
        self.distances, self.indices = self.index.search(self.windows, self.n_neighbors)
        self.select_shapelets()

    def get_window_label(self, window_id):
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        ts_id = window_id // number_of_windows_per_ts
        label = self.y[ts_id]
        return label

    def select_shapelets(self):
        self.selected = []
        for window_id in range(self.n_windows):
            label = self.get_window_label(window_id)
            same_label_cnt = 0
            for neighbor_id in self.indices[window_id]:
                n_label = self.get_window_label(neighbor_id)
                if n_label == label:
                    same_label_cnt += 1
            popularity = same_label_cnt / self.n_neighbors
            if popularity >= self.threshold:
                self.selected.append(window_id)

    def transform(self, X):
        windows = self._get_windows(X)
        shapelets = self.windows[self.selected]
        dists = cdist(windows, shapelets).reshape(X.shape[0], -1, len(self.selected))
        return dists.min(axis=1)
