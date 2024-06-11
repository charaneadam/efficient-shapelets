from src.data import Data
from src.experiments.helpers import transform_dataset, classify_dataset
from src.methods import SELECTION_METHODS
from src.exceptions import DataFailure


def run_dataset(dataset_name):
    dataset_info = dict()
    try:
        data = Data(dataset_name)
    except DataFailure as msg:
        with open("problematic_datasets.txt", "a") as f:
            f.write(f"Data Failure: {msg}\n")
        return

    for method_name in SELECTION_METHODS.keys():
        method_info = dict()
        X_tr, y_train, X_te, y_test = transform_dataset(
            data, method_name, {}, method_info
        )
        classify_dataset(X_tr, y_train, X_te, y_test, method_info)
        dataset_info[method_name] = method_info
    return dataset_info


def run():
    results = dict()
    datasets_names = ["CBF"]
    # datasets_names = ["DiatomSizeReduction"]
    for dataset_name in datasets_names:
        dataset_info = run_dataset(dataset_name)
        results[dataset_name] = dataset_info
    return results


if __name__ == "__main__":
    import json

    with open("comparisons.json", "w") as f:
        json.dump(run(), f, indent=2)
