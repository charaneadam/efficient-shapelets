from multiprocessing import Pool
from itertools import product
import json
from src.config import RESULTS_PATH
from src.storage.data import Data
from src.exceptions import (
    ClassificationFailure,
    DataFailure,
    TransformationFailrue,
)
from src.experiments.helpers import transform_dataset, classify_dataset

METHOD_NAME = "Kmeans"
results_path = RESULTS_PATH / "kmeans_parameters"
PROBLEMS_INFO = results_path / "problematic_datasets.txt"


def run_combination(data, params):
    info = dict()

    try:
        transformed = transform_dataset(data, METHOD_NAME, params, info)
        classify_dataset(*transformed, info)
    except TransformationFailrue:
        with open(PROBLEMS_INFO, "a") as f:
            f.write(f"Transformation problem: {data.dataset_name}\n")
    except ClassificationFailure:
        with open(PROBLEMS_INFO, "a") as f:
            f.write(f"Classification problem: {data.dataset_name}\n")
    except:
        with open(PROBLEMS_INFO, "a") as f:
            f.write(f"Unknown problem: {data.dataset_name}\n")

    return info


def run(dataset_name, _test=True):
    try:
        data = Data(dataset_name)
    except DataFailure:
        with open(PROBLEMS_INFO, "a") as f:
            f.write(f"Unreadable dataset: {dataset_name}\n")
        return

    if _test:
        lengths_percents = [p for p in range(10, 16, 5)]
        top_ks = [k for k in range(5, 11, 5)]
    else:
        lengths_percents = [p for p in range(10, 66, 5)]
        top_ks = [k for k in range(5, 150, 5)]

    list_params = params = product(lengths_percents, top_ks)
    list_params = [{"window_percentage": wl, "topk": k} for wl, k in params]
    res = []
    for params in list_params:
        res_dict = run_combination(data, params)
        if len(res_dict):
            res.append(res_dict)

    output_path = results_path / f"{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    datasets = ["CBF", "BME"]
    results_path.mkdir(parents=True, exist_ok=True)
    with Pool(2) as p:
        p.map(run, datasets)
