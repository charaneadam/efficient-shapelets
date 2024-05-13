from multiprocessing import Pool
from itertools import product
import json
import numpy as np
from src.data import Data
from src.exceptions import DatasetUnreadable
from src.experiments.helpers import transform_dataset, classify_dataset

METHOD_NAME = "Kmeans"


def run_combination(data, params):
    info = dict()
    X_tr, y_tr, X_te, y_te = transform_dataset(data, METHOD_NAME, params, info)
    classify_dataset(X_tr, y_tr, X_te, y_te, info)
    return info


def run(dataset_name):
    try:
        data = Data(dataset_name)
    except DatasetUnreadable:
        with open("problematic_datasets.txt", "a") as f:
            f.write(f"Unreadable dataset: {dataset_name}\n")
        return

    _, ts_length = data.X_train.shape
    # lengths_percents = [p for p in np.arange(0.1, 0.66, 0.05)]
    lengths_percents = [p for p in np.arange(0.1, 0.2, 0.05)]

    # top_ks = [k for k in range(5, 100, 5)]
    top_ks = [k for k in range(5, 10, 5)]

    list_params = params = product(lengths_percents, top_ks)
    list_params = [{"window_percentage": wl, "topk": k} for wl, k in params]
    res = [run_combination(data, params) for params in list_params]
    with open(f"{dataset_name}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    # datasets = ["CBF", "GunPoint", "ArrowHead", "Beef", "BME"]
    datasets = ["CBF"]
    with Pool(4) as p:
        p.map(run, datasets)
