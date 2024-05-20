import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import faiss


class NearestNeighborTransform:
    def __init__(self, verbose=0, n_neighbors=10, window_size=30):
        self.window_size: int | None = None
        self.index = None
        self.X = None
        self.y = None
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.df = None

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
        self.topk = 10

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

    def get_window_ts(self, window_id):
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        ts_id = window_id // number_of_windows_per_ts
        return ts_id

    def get_window_label(self, window_id):
        ts_id = self.get_window_ts(window_id)
        label = self.y[ts_id]
        return label

    def select_shapelets(self):
        labels = np.apply_along_axis(
            self.get_window_label, 0, np.arange(self.n_windows)
        )
        covered = np.apply_along_axis(self.get_window_ts, 0, self.indices)

        indices_labels = labels[self.indices]
        same = indices_labels == indices_labels[:, 0].reshape(-1, 1)

        popularity = same.sum(axis=1)

        distances = (self.distances * same).sum(axis=1) / popularity

        get_c_f = lambda window_id: (covered[window_id], same[window_id])
        get_n_covered = lambda c, s: len(set(c[s]))
        f = lambda window_id: get_n_covered(*get_c_f(window_id))
        n_covered_per_window = [f(i) for i in range(self.n_windows)]

        df_content = np.array([labels, n_covered_per_window, popularity, distances]).T
        df_columns = ["label", "n_covered", "popularity", "distance"]
        df = pd.DataFrame(df_content, columns=df_columns)
        df = df.astype({"label": "int32", "popularity": "int32", "n_covered": "int32"})
        sort_order = ["n_covered", "popularity", "distance"]
        asc = [False, False, True]
        df.sort_values(by=sort_order, ascending=asc, inplace=True)
        self.df = df

    def transform(self, X, k=None):
        if k is None:
            k = self.k
        self.selected = []
        lbls = set(self.y)
        for lbl in lbls:
            self.selected.extend(list(self.df[self.df.label == lbl].index[:k]))
        windows = self._get_windows(X)
        shapelets = self.windows[self.selected]
        dists = cdist(windows, shapelets).reshape(X.shape[0], -1, len(self.selected))
        return dists.min(axis=1)
