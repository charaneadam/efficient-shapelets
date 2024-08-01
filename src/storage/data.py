import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from src.exceptions import DataFailure
from src.config import DATA_PATH
from src.storage.database import BaseModel
from peewee import CharField, IntegerField, BooleanField


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


class Windows:
    def __init__(self, window_size: int, skip_size=None):
        self.size: int = window_size
        self.skip: int
        self.windows_per_ts: int
        if skip_size is None:
            self.skip = int(0.25 * window_size)
        else:
            self.skip = skip_size

    def get_windows(self, X):
        res = sliding_window_view(X, window_shape=(1, self.size)).squeeze()
        if self.skip > 1:
            ts_length = X.shape[1]
            # Indices of the windows that we keep from all the sliding windows.
            selected = np.arange(0, ts_length - self.size + 1, self.skip)
            res = res[:, selected, :]
        # Windows have shape: (n_ts, n_windows, window_size)
        self.windows_per_ts = res.shape[1]

        # Below we reshape the array such that the shape becomes
        # (n_ts * n_windows, window_size). This is the format that
        # Kmeans and nearest neighbor algorithms expect.
        res = res.reshape(-1, self.size)

        # Z-normalize the windows
        res = StandardScaler().fit_transform(res.T).T

        return res.astype("float32")

    def get_ts_index_of_window(self, window_index):
        # This method returns the index of the time series from which a
        # window was extracted given its index. (indexing is 0-based ofc)
        return window_index // self.windows_per_ts


class Data:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X_train, self.y_train = get_dataset(dataset_name, train=True)
        self.X_test, self.y_test = get_dataset(dataset_name, train=False)
        self.n_ts, self.ts_length = self.X_train.shape
        self.labels = sorted(np.unique(self.y_test))


class Dataset(BaseModel):
    name = CharField(unique=True)
    data_type = CharField()
    train = IntegerField()
    test = IntegerField()
    n_classes = IntegerField()
    length = IntegerField()
    missing_values = BooleanField(default=False)
    problematic = BooleanField(default=False)


def init_ucr_metadata(db, mark_ts_with_nan=True):
    import pandas as pd

    with db:
        Dataset.create_table()

    df = pd.read_csv(
        "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
    )
    df = df[df.Length != "Vary"]
    cols = ["Type", "Name", "Train ", "Test ", "Class", "Length"]
    names = ["data_type", "name", "train", "test", "n_classes", "length"]
    for row in df[cols].values:
        row[-1] = int(row[-1])
        dataset = Dataset.create(**dict(zip(names, row)))
        if mark_ts_with_nan:
            try:
                get_dataset(dataset.name, train=True)
                get_dataset(dataset.name, train=True)
            except DataFailure:
                dataset.missing_values = True
                dataset.save()


def get_datasets_info():
    import pandas as pd
    from src.storage.database import paper_engine

    query = """SELECT id, name, train, test, n_classes, length 
                FROM dataset ORDER BY train*length"""
    return pd.read_sql(query, paper_engine)
