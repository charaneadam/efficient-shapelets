import warnings
import pandas as pd

from src.benchmarks.classification.utils import _classify, transform

from src.storage.database import engine
from sqlalchemy import inspect
from src.storage.data import Data
from src.classifiers import CLASSIFIERS_NAMES
from src.storage.database import (
    SAME_LENGTH_CLASSIFICATION_TABLE_NAME,
    SAME_LENGTH_CANDIDATES_TABLE_NAME,
)


def _select_best_k(df, label, method, K, data):
    view = df[df.label == label]
    info = view.sort_values(by=method, ascending=False)[
        ["ts_id", "start", "length"]
    ].values[:K]
    shapelets = []
    for ts_id, start, length in info:
        cand = data.X_train[ts_id][start : start + length]
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


def compare(dataset_name, window_length):
    warnings.simplefilter("ignore")
    data = Data(dataset_name)
    df = pd.read_sql(
        f"""SELECT * FROM {SAME_LENGTH_CANDIDATES_TABLE_NAME}
        WHERE dataset='{dataset_name}' AND length={window_length}""",
        engine,
    )
    results = []
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
        f"SELECT DISTINCT dataset_id FROM {SAME_LENGTH_CANDIDATES_TABLE_NAME}", engine
    ).values.squeeze()

    inspector = inspect(engine)
    if inspector.has_table(SAME_LENGTH_CLASSIFICATION_TABLE_NAME):
        current_df = pd.read_sql(SAME_LENGTH_CLASSIFICATION_TABLE_NAME, engine)
        computed = set(current_df.dataset_id.unique())
    else:
        computed = set()

    columns = ["dataset_id", "method", "K_shapelets"] + CLASSIFIERS_NAMES
    datasets = [7]
    for dataset_id in datasets:
        if dataset_id in computed:
            continue

        dataset_name = str(
            pd.read_sql(
                f"SELECT name FROM dataset WHERE id={dataset_id}", engine
            ).values.squeeze()
        )
        try:
            lengths = pd.read_sql(
                f"""SELECT DISTINCT length
                FROM {SAME_LENGTH_CANDIDATES_TABLE_NAME}
                WHERE dataset='{dataset_id}'""",
                engine,
            ).values.squeeze()
            for window_size in lengths:
                results = compare(dataset_name, window_size)
                df = pd.DataFrame(results, columns=columns)
                df["window_size"] = window_size
                df.to_sql(
                    SAME_LENGTH_CLASSIFICATION_TABLE_NAME,
                    engine,
                    if_exists="append",
                    index=False,
                )
        except:
            print(f"Error happened with dataset {dataset_name}")


if __name__ == "__main__":
    run()
