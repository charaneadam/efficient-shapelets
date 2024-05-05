import os
from pathlib import Path
import numpy as np
from aeon.datasets import load_from_tsfile
from sklearn.metrics import accuracy_score


def get_dataset(
    dataset_name,
    train,
) -> tuple[np.ndarray, np.ndarray]:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = Path(BASE_DIR).parent / "data"
    split = "TRAIN" if train else "TEST"
    filepath = str(DATA_PATH / f"{dataset_name}/{dataset_name}_{split}.ts")
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X_train, self.y_train = get_dataset(dataset_name, train=True)
        self.X_test, self.y_test = get_dataset(dataset_name, train=False)


def transform(data, Approach, params={}):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test
    app = Approach(**params)
    app.fit(X_train, y_train)

    X_tr = app.transform(X_train)
    X_te = app.transform(X_test)
    return X_tr, y_train, X_te, y_test


def evaluate(X_tr, y_tr, X_te, y_te, Model, params={}):
    model = Model(**params)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_pred, y_te)
    print(f"Accuracy: {acc}")
