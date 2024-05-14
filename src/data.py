import numpy as np
import pandas as pd

from src.exceptions import DatasetUnreadable

from .config import DATA_PATH, METADATA_PATH


def get_dataset(
    dataset_name,
    train,
) -> tuple[np.ndarray, np.ndarray]:
    split = "TRAIN" if train else "TEST"
    filepath = str(DATA_PATH / f"{dataset_name}/{dataset_name}_{split}.tsv")
    data = np.genfromtxt(filepath, delimiter="\t")
    x, y = data[:, 1:], data[:, 0].astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name):
        try:
            self.dataset_name = dataset_name
            self.X_train, self.y_train = get_dataset(dataset_name, train=True)
            self.X_test, self.y_test = get_dataset(dataset_name, train=False)
        except:
            raise DatasetUnreadable(dataset_name)


def get_metadata():
    df = pd.read_csv(METADATA_PATH)
    df.set_index("ID", inplace=True)
    same_length_datasets = ~(df["Length"] == "Vary")
    df = df[same_length_datasets]
    df.loc[:, "Length"] = df["Length"].astype(int)
    df.rename(columns={"Train ": "Train"}, inplace=True)
    df["train_size"] = df["Train"] * df["Length"]
    df.sort_values(by="train_size", inplace=True)
    return df
