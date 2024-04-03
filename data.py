from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from aeon.datasets import load_from_tsfile
import numpy as np


def get_dataset(dataset_name, train, path) -> tuple[np.ndarray, np.ndarray]:
    split = "TRAIN" if train else "TEST"
    filepath = f"{path}/{dataset_name}/{dataset_name}_{split}.ts"
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name, path="data"):
        x, y = get_dataset(dataset_name=dataset_name, train=True, path=path)
        self.x_train: np.ndarray = x
        self.y_train: np.ndarray = y
        self.n_ts = self.x_train.shape[0]
        self.windows = None
        self.window_size: int = 0
        self.clusters: np.ndarray

    def get_sliding_windows(self, window_length=25, normalize=True):
        if self.window_size != window_length:
            self.window_size = window_length
            self.windows = sliding_window_view(
                self.x_train, window_shape=window_length, axis=1
            ).reshape(-1, window_length)
            if normalize:
                self.windows = StandardScaler().fit_transform(self.windows.T).T
        return self.windows
