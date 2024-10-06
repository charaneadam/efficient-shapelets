import numpy as np

from src.exceptions import DataFailure
from src.config import DATA_PATH


def get_dataset(
    dataset_name,
    train,
) -> tuple[np.ndarray, np.ndarray]:
    split = "TRAIN" if train else "TEST"
    filepath = str(DATA_PATH / f"{dataset_name}/{dataset_name}_{split}.tsv")
    try:
        data = np.genfromtxt(filepath, delimiter="\t")
    except:
        raise DataFailure("Cannot open data file.")
    x, y = data[:, 1:], data[:, 0].astype(int)
    if np.isnan(x).any() or np.isnan(y).any():
        raise DataFailure(
            f"{split.capitalize()} split of {dataset_name} has \
        missing values."
        )
    return x, y


class Data:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X_train, self.y_train = get_dataset(dataset_name, train=True)
        self.X_test, self.y_test = get_dataset(dataset_name, train=False)
        self.n_ts, self.ts_length = self.X_train.shape
        self.labels = sorted(np.unique(self.y_test))
