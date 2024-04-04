from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from aeon.datasets import load_from_tsfile
import numpy as np


def get_dataset(
    dataset_name,
    train,
) -> tuple[np.ndarray, np.ndarray]:
    split = "TRAIN" if train else "TEST"
    filepath = f"../data/{dataset_name}/{dataset_name}_{split}.ts"
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name, window_size):
        x, y = get_dataset(dataset_name=dataset_name, train=True)
        self.x_train: np.ndarray = x
        self.y_train: np.ndarray = y
        self.n_ts = self.x_train.shape[0]
        self.window_size: int = window_size
        self.windows = sliding_window_view(
            self.x_train, window_shape=self.window_size, axis=1
        ).reshape(-1, self.window_size)

    def get_sliding_windows(self, normalize=True):
        self.windows = self.windows
        if normalize:
            self.windows = StandardScaler().fit_transform(self.windows.T).T
        return self.windows
