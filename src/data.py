import numpy as np
from aeon.datasets import load_from_tsfile

from src.exceptions import DatasetUnreadable

from .config import DATA_PATH


def get_dataset(
    dataset_name,
    train,
) -> tuple[np.ndarray, np.ndarray]:
    split = "TRAIN" if train else "TEST"
    filepath = str(DATA_PATH / f"{dataset_name}/{dataset_name}_{split}.ts")
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name):
        try:
            self.dataset_name = dataset_name
            self.X_train, self.y_train = get_dataset(dataset_name, train=True)
            self.X_test, self.y_test = get_dataset(dataset_name, train=False)
        except:
            raise DatasetUnreadable(dataset_name)
