import numpy as np
import pandas as pd
from src.storage.database import fix_engine, paper_engine

from src.storage.database import fix_engine
from src.storage.data import Data
from src.benchmarks.windows_evaluation.utils import distance_numba
from src.benchmarks.classification.utils import _classify
from src.exceptions import NumberShapeletsFailure


def get_datasets():
    query = """SELECT DISTINCT dataset FROM candidates
                WHERE id IN (SELECT DISTINCT candidate FROM evaluation);"""
    evaluated_datasets = pd.read_sql(query, fix_engine)
    dataset_ids = ", ".join(map(str, evaluated_datasets.dataset.values))
    query = f"""SELECT id, name FROM dataset
                WHERE id IN ({dataset_ids})
                ORDER BY train*test*length"""
    df = pd.read_sql(query, paper_engine)
    ids = df.id.values
    names = df.name.values
    return ids, names


def get_candidate_ids(dataset_id, method_id):
    query = f"""SELECT id FROM candidates
                WHERE dataset={dataset_id} AND method={method_id} """
    return pd.read_sql(query, fix_engine).id.values


def shapelet_transform(X, candidates):
    transformed = []
    for candidate in candidates:
        candidate_transform = []
        for x in X:
            candidate_transform.append(distance_numba(x, candidate))
        transformed.append(candidate_transform)
    return np.array(transformed).T


def candidates_info(dataset_id, method_id):
    candidates_ids = ", ".join(map(str, get_candidate_ids(dataset_id, method_id)))
    candidates = pd.read_sql(
        f"SELECT * FROM candidates WHERE id IN ({candidates_ids})",
        fix_engine,
    )
    evaluations = pd.read_sql(
        f"SELECT * FROM evaluation WHERE candidate IN ({candidates_ids})", fix_engine
    ).rename(columns={"candidate": "id"})
    df = pd.merge(candidates, evaluations, on="id")
    return df


def get_candidates(data, candidates_info, evaluation_method_name="silhouette"):
    K_shapelets = 100
    candidates = []
    less_than_100 = False
    for label in candidates_info.label.unique():
        view = candidates_info[candidates_info.label == label]
        view = view.sort_values(evaluation_method_name, ascending=False).iloc[
            :K_shapelets, :
        ]
        if view.shape[0] < 100:
            less_than_100 = True
            raise NumberShapeletsFailure
        for ts_id, start, end in view[["ts", "first", "last"]].values:
            candidates.append(data.X_train[ts_id][start:end])
    return candidates


def transform(data, candidates_info, method_name):
    try:
        candidates = get_candidates(data, candidates_info, method_name)
    except NumberShapeletsFailure as e:
        with open("classification_errors.log", "a") as f:
            f.write(f"dataset {dataset_name} missing.\n")
        return None, None
    X_tr = shapelet_transform(data.X_train, candidates)
    X_te = shapelet_transform(data.X_test, candidates)
    return X_tr, X_te


def classify(X_tr, y_tr, X_te, y_te):
    n_labels = len(set(y_tr))
    K_shapelets = X_tr.shape[1] // n_labels
    starts = np.arange(n_labels) * K_shapelets
    results = []
    for k in [5, 10, 25, 50, 75, 100]:
        if k > K_shapelets:
            continue
        Xtr = np.concatenate([X_tr[:, s : s + k] for s in starts], axis=1)
        Xte = np.concatenate([X_te[:, s : s + k] for s in starts], axis=1)
        fit_time, predict_time, acc, f1, labels, precision, recall = _classify(
            "Logistic Regression", Xtr, y_tr, Xte, y_te
        )
        results.append([k, acc, f1, fit_time, predict_time])
    return results


def classify_dataset(dataset_id, dataset_name):
    data = Data(dataset_name)
    for extraction_methond_id in [6, 7, 8]:
        info = candidates_info(dataset_id, extraction_methond_id)
        for evaluation_method in [
            "silhouette",
            "fstat",
            "binary info",
            "multiclass info",
        ]:
            X_tr, X_te = transform(data, info, evaluation_method)
            if X_tr is None or X_te is None:
                continue
            results = classify(X_tr, data.y_train, X_te, data.y_test)
            results = pd.DataFrame(
                results, columns=["k", "accuracy", "f1", "fit time", "predict time"]
            )
            results["dataset"] = dataset_id
            results["evaluation"] = evaluation_method
            results["extraction"] = extraction_methond_id
            results.to_sql(
                "classification", fix_engine, if_exists="append", index=False
            )


if __name__ == "__main__":
    dataset_ids, dataset_names = get_datasets()
    for dataset_id, dataset_name in zip(dataset_ids, dataset_names):
        classify_dataset(dataset_id, dataset_name)
