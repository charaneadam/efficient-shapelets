import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from aeon.datasets import load_from_tsfile
import numpy as np
from scipy.spatial.distance import cdist


def get_dataset(
    dataset_name,
    train,
) -> tuple[np.ndarray, np.ndarray]:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = Path(BASE_DIR) / "data"
    split = "TRAIN" if train else "TEST"
    filepath = str(DATA_PATH / f"{dataset_name}/{dataset_name}_{split}.ts")
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class TrainData:
    """
    A class to store a dataset, and returns some info:
    - get sliding windows of the data
    - return the label of a window given its index
    - Given a list of window indices return their corresponding TS and labels
    - Given a list of windows indices return the shapelet transform
    """

    def __init__(self, dataset_name, window_size, preprocessors=None):
        x, y = get_dataset(dataset_name=dataset_name, train=True)
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.n_ts = self.x.shape[0]
        self.ts_length = self.x.shape[1]
        self.window_size: int = window_size
        self.windows = sliding_window_view(
            self.x, window_shape=self.window_size, axis=1
        ).reshape(-1, self.window_size)
        self.windows = StandardScaler().fit_transform(self.windows.T).T
        if preprocessors:
            for processor in preprocessors:
                self.windows = processor.fit_transform(self.windows)
        self._ts_covered: dict | None = None

    def get_sliding_windows(self):
        return self.windows

    def get_window_label(self, window_id):
        if self._ts_covered is None:
            self._ts_covered = {}
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        ts_id = window_id // number_of_windows_per_ts
        label = self.y[ts_id]
        if label in self._ts_covered:
            set_ts = self._ts_covered[label]
            set_ts.add(ts_id)
        else:
            set_ts = {ts_id}
        self._ts_covered[label] = set_ts
        return label

    def windows_labels_and_covered_ts(self, windows_ids):
        windows_labels = [self.get_window_label(wid) for wid in windows_ids]
        return np.array(windows_labels), self._ts_covered


class TestData:
    def __init__(self, dataset_name, window_size):
        x, y = get_dataset(dataset_name=dataset_name, train=False)
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.n_ts = self.x.shape[0]
        self.ts_length = self.x.shape[1]
        self.windows = sliding_window_view(
            self.x, window_shape=window_size, axis=1
        ).reshape(-1, window_size)
        self.windows = StandardScaler().fit_transform(self.windows.T).T


class Data:
    """
    A class to store a dataset, and returns some info:
    - get sliding windows of the data
    - return the label of a window given its index
    - Given a list of window indices return their corresponding TS and labels
    - Given a list of windows indices return the shapelet transform
    """

    def __init__(self, dataset_name, window_size, preprocessors=None):
        self.window_size = window_size
        self._train = TrainData(dataset_name, window_size, preprocessors)
        self._test = TestData(dataset_name, window_size)

    def unique_labels(self):
        return np.unique(self._train.y)

    def get_sliding_windows(self):
        return self._train.windows

    def get_window_label(self, window_id):
        return self._train.get_window_label(window_id)

    def windows_labels_and_covered_ts(self, windows_ids):
        return self._train.windows_labels_and_covered_ts(windows_ids)

    def shapelet_transform(self, windows_ids, train=True):
        candidates = self._train.windows[windows_ids]
        if train:
            windows = self._train.windows
            n_ts = self._train.n_ts
        else:
            windows = self._test.windows
            n_ts = self._test.n_ts
        window_size = self._train.window_size

        windows = windows.reshape(n_ts, -1, window_size)
        res = [cdist(candidates, windows).min(axis=1) for windows in windows]
        return np.array(res)
