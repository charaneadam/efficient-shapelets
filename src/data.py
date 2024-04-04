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
        self.ts_length = self.x_train.shape[1]
        self.window_size: int = window_size
        self.windows = sliding_window_view(
            self.x_train, window_shape=self.window_size, axis=1
        ).reshape(-1, self.window_size)
        self.windows = StandardScaler().fit_transform(self.windows.T).T
        self._ts_covered: dict

    def get_sliding_windows(self):
        return self.windows

    def _get_window_label(self, window_id):
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        ts_id = window_id // number_of_windows_per_ts
        label = self.y_train[ts_id]
        if label in self._ts_covered:
            set_ts = self._ts_covered[label]
            set_ts.add(ts_id)
        else:
            set_ts = {ts_id}
        self._ts_covered[label] = set_ts
        return label

    def windows_labels_and_covered_ts(self, windows_ids):
        self._ts_covered = {}
        windows_labels = [self._get_window_label(wid) for wid in windows_ids]
        return np.array(windows_labels), self._ts_covered
