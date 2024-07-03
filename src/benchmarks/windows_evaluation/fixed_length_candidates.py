import numpy as np
import pandas as pd
from sqlalchemy import inspect
from src.benchmarks.get_experiment import get_datasets
from src.benchmarks.windows_evaluation.bruteforce import evaluate
from src.exceptions import NormalizationFailure
from src.storage.data import Data
from src.storage.database import engine

from src.storage.database import SAME_LENGTH_CANDIDATES_TABLE_NAME


def sample_subsequence_positions(ts_length, window_length):
    start_pos = np.random.randint(ts_length - window_length + 1)
    end_pos = start_pos + window_length
    return start_pos, end_pos


def normalize(candidate):
    try:
        return (candidate - np.mean(candidate)) / np.std(candidate)
    except:
        raise NormalizationFailure


def candidates_and_tsids(data, window_length):
    """Given some data, this function samples a number of shapelets with
    different lengths. It returns the Z-normalized candidates as well as
    the ids of the time series from which they have been extracted."""
    ts_length = data.ts_length
    n_shapelets = max(300, int(0.2 * ts_length))
    labels = list(set(data.y_train))
    candidates = []
    ids = []
    fail = 0
    for label in labels:
        ts_ids = np.where(data.y_train == label)[0]
        remaining = n_shapelets
        fail = 0
        while remaining > 0 and fail < 10:
            try:
                ts_id = np.random.choice(ts_ids)
                start_pos, end_pos = sample_subsequence_positions(
                    ts_length, window_length
                )
                candidate = data.X_train[ts_id][start_pos:end_pos]
                # candidate = normalize(candidate)
                ids.append([ts_id, start_pos, window_length])
                remaining -= 1
                candidates.append(candidate)
            except NormalizationFailure:
                """Failure sometimes happen when normalizing a candidate due to
                numeric calculations failure (most of the time the standard
                deviation is 0. When a failure happens, we simply retry
                with a new sample, and keep repeating this process till
                the number of candidates is satisfied"""
                fail += 1
    if fail == 10:
        return
    candidates_info = np.array(ids)
    df = evaluate(data, candidates, candidates_info[:, 0])
    df["dataset"] = data.dataset_name
    df["ts_id"] = candidates_info[:, 0]
    df["start"] = candidates_info[:, 1]
    df["length"] = candidates_info[:, 2]
    df.to_sql(SAME_LENGTH_CANDIDATES_TABLE_NAME, engine, if_exists="append", index=False)


def run():
    datasets = get_datasets()
    inspector = inspect(engine)
    processed_datasets = dict()
    if inspector.has_table(SAME_LENGTH_CANDIDATES_TABLE_NAME):
        df = pd.read_sql(SAME_LENGTH_CANDIDATES_TABLE_NAME, engine).groupby(["dataset"])["length"]
        processed_datasets = df.agg("unique").to_dict()

    for dataset in datasets:
        data = None
        error = False
        for window_perc in [0.05, 0.1, 0.2, 0.3, 0.5, 0.6]:
            if error:
                continue
            window_size = int(window_perc * dataset.length)
            if (
                dataset.name in processed_datasets
                and window_size in processed_datasets[dataset.name]
            ) or dataset.length < 60:
                continue
            if data is None:
                data = Data(dataset.name)
            try:
                candidates_and_tsids(data, window_size)
            except:
                error = True
                print(f"Error with {dataset.name} length {window_size}")


if __name__ == "__main__":
    run()
