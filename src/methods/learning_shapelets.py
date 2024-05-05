import numpy as np
from scipy.spatial.distance import cdist
from numpy.lib.stride_tricks import sliding_window_view

from pyts.classification import LearningShapelets


class LearningShapeletsTransform:
    def __init__(self):
        self.clf = LearningShapelets(random_state=42, tol=0.01)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def transform(self, X):
        res = []
        for shapelet in self.clf.shapelets_[0]:
            window_size = len(shapelet)
            windows = sliding_window_view(X, window_shape=window_size, axis=1)
            windows = windows.reshape(-1, window_size)
            ts_dists = (
                cdist(shapelet.reshape(1, -1), windows)
                .reshape(X.shape[0], -1)
                .min(axis=1)
            )
            res.append(ts_dists)
        return np.array(res).T
