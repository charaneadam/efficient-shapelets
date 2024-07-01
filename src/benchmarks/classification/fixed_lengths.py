import warnings
from numba import set_num_threads
import numpy as np
import pandas as pd

from src.benchmarks.get_experiment import get_datasets
from src.benchmarks.windows_evaluation.bruteforce import _eval_bruteforce
from src.benchmarks.classification.utils import _classify, transform

from src.storage.database import engine
from sqlalchemy import inspect
from src.classifiers import CLASSIFIERS_NAMES
from src.storage.data import Data


def sample_subsequence_positions(ts_length, window_length):
    start_pos = np.random.randint(ts_length - window_length + 1)
    end_pos = start_pos + window_length
    return start_pos, end_pos


def candidates_and_tsids(data, window_length):
    """Given some data, this function samples a number of shapelets with
    different lengths. It returns the Z-normalized candidates as well as
    the ids of the time series from which they have been extracted."""
    ts_length = data.ts_length
    n_shapelets = max(300, int(0.2 * ts_length))
    labels = list(set(data.y_train))
    candidates = []
    ids = []
    positions = []
    for label in labels:
        ts_ids = np.where(data.y_train == label)[0]
        remaining = n_shapelets
        while remaining > 0:
            try:
                ts_id = np.random.choice(ts_ids)
                start_pos, end_pos = sample_subsequence_positions(
                    ts_length, window_length
                )
                candidate = data.X_train[ts_id][start_pos:end_pos]
                candidate = (candidate - np.mean(candidate)) / np.std(candidate)
                positions.append([data.dataset_name, ts_id, start_pos, window_length])
                remaining -= 1
                candidates.append(candidate)
                ids.append(ts_id)
            except:
                """Failure sometimes happen when normalizing a candidate due to
                numeric calculations failure (most of the time the standard
                deviation is 0. When a failure happens, we simply retry
                with a new sample, and keep repeating this process till
                the number of candidates is satisfied"""
                pass
    df = pd.DataFrame(positions, columns=["dataset", "ts_id", "start", "length"])
    df.to_sql("fixed_lengths_candidates", engine, if_exists="append", index=False)
    return candidates, ids


def evaluate(data, windows, windows_ts_ids):
    """Given data, windows and their corresponding ts_ids from which they have
    been extracted; this function return a dataframe with the scores,
    their timings, as well as the label of each window"""
    results = _eval_bruteforce(data.X_train, data.y_train, windows, windows_ts_ids)
    cols = ["silhouette", "silhouette_time", "fstat", "fstat_time", "gain", "gain_time"]
    df = pd.DataFrame(results, columns=cols)
    windows_labels = data.y_train[windows_ts_ids]
    df["label"] = windows_labels
    return df


def _select_best_k(df, label, method, K):
    """Returns the best *K* shapelets for representing time series labelled
    *label* using some *method* ("silhouette", "gain" or "fstat")"""
    view = df[df.label == label]
    return view.sort_values(by=method, ascending=False).index[:K].values


def classify(df, windows, data, method, k):
    shapelets = []
    labels = list(set(data.y_train))
    for label in labels:
        indices = _select_best_k(df, label, method, k)
        shapelets.extend([windows[index] for index in indices])
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
    candidates, candidatests_ids = candidates_and_tsids(data, window_length)
    df = evaluate(data, candidates, candidatests_ids)
    results = []
    for method in ["silhouette", "gain", "fstat"]:
        for K in [3, 5, 10, 20, 50, 100]:
            accuracies = classify(df, candidates, data, method, K)
            models_accuracies = [
                accuracies.get(clf_name, None) for clf_name in CLASSIFIERS_NAMES
            ]
            result = [dataset_name, method, K] + models_accuracies
            results.append(result)
    return results


def run():
    datasets = get_datasets()
    columns = ["dataset", "method", "K_shapelets"] + CLASSIFIERS_NAMES

    inspector = inspect(engine)
    TABLE_NAME = "fixed_lengths"
    if inspector.has_table(TABLE_NAME):
        current_df = pd.read_sql(TABLE_NAME, engine)
        computed = set(current_df.dataset.unique())
    else:
        computed = set()
    for dataset in datasets:
        if dataset.length < 60 or dataset in computed:
            continue
        try:
            for window_perc in [0.05, 0.1, 0.2, 0.3, 0.5, 0.6]:
                window_size = int(window_perc * dataset.length)
                results = compare(dataset.name, window_size)
                df = pd.DataFrame(results, columns=columns)
                df["window_size"] = window_size
                df.to_sql(TABLE_NAME , engine, if_exists="append", index=False)
        except:
            print(f"Error happened with dataset {dataset.name}")


if __name__ == "__main__":
    set_num_threads(4)
    run()
