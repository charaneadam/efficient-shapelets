from multiprocessing import Pool
from itertools import product
import json
from src.config import RESULTS_PATH
from src.data import Data, get_metadata
from src.exceptions import DatasetUnreadable
from src.experiments.helpers import transform_dataset, classify_dataset

METHOD_NAME = "Kmeans"
results_path = RESULTS_PATH / "kmeans_parameters"
PROBLEMS_INFO = results_path / "problematic_datasets.txt"


def run_combination(data, params):
    info = dict()
    X_tr, y_tr, X_te, y_te = transform_dataset(data, METHOD_NAME, params, info)
    classify_dataset(X_tr, y_tr, X_te, y_te, info)
    return info


def run(dataset_name):
    try:
        data = Data(dataset_name)
    except DatasetUnreadable:
        with open(PROBLEMS_INFO, "a") as f:
            f.write(f"Unreadable dataset: {dataset_name}\n")
        return

    # Small combination of parameters for testing
    lengths_percents = [p for p in range(10, 66, 5)]
    top_ks = [k for k in range(5, 150, 5)]

    list_params = params = product(lengths_percents, top_ks)
    list_params = [{"window_percentage": wl, "topk": k} for wl, k in params]
    res = [run_combination(data, params) for params in list_params]

    output_path = results_path / f"{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    datasets_metadata = get_metadata()
    datasets = datasets_metadata.Name.values
    results_path.mkdir(parents=True, exist_ok=True)
    with Pool(32) as p:
        p.map(run, datasets)
