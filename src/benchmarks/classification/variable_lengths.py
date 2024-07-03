import warnings
import pandas as pd

from src.benchmarks.classification.utils import _classify, transform

from src.storage.database import engine
from sqlalchemy import inspect
from src.classifiers import CLASSIFIERS_NAMES
from src.storage.data import Data

CLASSIFICATION_VARIABLE_TABLE_NAME = "classification_variable_lengths"


def _select_best_k(df, label, method, K, data):
    view = df[df.label == label]
    info = view.sort_values(by=method, ascending=False)[
        ["ts_id", "start", "end"]
    ].values[:K]
    shapelets = []
    for ts_id, start, end in info:
        cand = data.X_train[ts_id][start:end]
        cand = (cand - cand.mean()) / cand.std()
        shapelets.append(cand)
    return shapelets


def classify(df, data, method, k):
    shapelets = []
    labels = list(set(data.y_train))
    for label in labels:
        candidates = _select_best_k(df, label, method, k, data)
        shapelets.extend(candidates)
    X_tr, X_te = transform(data, shapelets)
    accuracies = {}
    for clf_name in CLASSIFIERS_NAMES:
        res = _classify(clf_name, X_tr, data.y_train, X_te, data.y_test)
        fit_time, predict_time, acc, f1, labels, precision, recall = res
        accuracies[clf_name] = acc
    return accuracies


def compare(dataset_name):
    warnings.simplefilter("ignore")
    data = Data(dataset_name)
    results = []
    df = pd.read_sql(
        f"""SELECT * FROM random_lengths_candidates
        WHERE dataset='{dataset_name}'""",
        engine,
    )
    for method in ["silhouette", "gain", "fstat"]:
        for K in [3, 5, 10, 20, 50, 100]:
            accuracies = classify(df, data, method, K)
            models_accuracies = [
                accuracies.get(clf_name, None) for clf_name in CLASSIFIERS_NAMES
            ]
            result = [dataset_name, method, K] + models_accuracies
            results.append(result)
    return results


def run():
    datasets = pd.read_sql(
        "SELECT DISTINCT dataset FROM random_lengths_candidates", engine
    ).values.squeeze()
    columns = ["dataset", "method", "K_shapelets"] + CLASSIFIERS_NAMES
    inspector = inspect(engine)
    if inspector.has_table(CLASSIFICATION_VARIABLE_TABLE_NAME):
        current_df = pd.read_sql(CLASSIFICATION_VARIABLE_TABLE_NAME, engine)
        computed = set(current_df.dataset.unique())
    else:
        computed = set()
    for dataset in datasets:
        if dataset in computed:
            continue
        try:
            results = compare(dataset)
            df = pd.DataFrame(results, columns=columns)
            df.to_sql(
                CLASSIFICATION_VARIABLE_TABLE_NAME,
                engine,
                if_exists="append",
                index=False,
            )
        except Exception as e:
            print(f"Error during classification of dataset {dataset}.")
            print(f"Error: {e}")


if __name__ == "__main__":
    run()
