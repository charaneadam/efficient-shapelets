import numpy as np
import pandas as pd
from sqlalchemy import inspect
from src.benchmarks.get_experiment import get_datasets
from src.storage.data import Data
from src.storage.database import engine
from src.benchmarks.windows_evaluation.bruteforce import evaluate

TABLE_NAME = "random_lengths_candidates"


def sample_subsequence_positions(ts_length):
    """
    Given the length of a time series, this function return a candidate s.t:
    - Starting position is in the range [0, index_of(90% of the ts_length)]
    - The length is in the range of [5% of ts_length, 70% of ts_length]
    - We set the length to a max of (3, and length). This is for the case
        when we take, say 5% of a really small time series. eg. 5% of 12
        is 0.6, which will be rounded to 0
    - Ending position is simply: start_pos + length (in case end pos
        is out of bound, we set it to the last timestamp)
    """
    start_pos = np.random.randint(int(0.9 * ts_length))
    length = np.random.randint(int(0.05 * ts_length), int(0.7 * ts_length))
    length = max(length, 3)
    end_pos = min(ts_length, start_pos + length)
    return start_pos, end_pos


def candidates_and_tsids(data):
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
                start_pos, end_pos = sample_subsequence_positions(ts_length)
                candidate = data.X_train[ts_id][start_pos:end_pos]
                candidate = (candidate - np.mean(candidate)) / np.std(candidate)
                remaining -= 1
                candidates.append(candidate)
                ids.append([ts_id, start_pos, end_pos])
            except:
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
    df["end"] = candidates_info[:, 2]
    df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)


def run():
    datasets = get_datasets()
    inspector = inspect(engine)
    computed = {}
    if inspector.has_table(TABLE_NAME):
        current_df = pd.read_sql(TABLE_NAME, engine)
        computed = set(current_df.dataset.unique())
    for dataset in datasets:
        if dataset.length < 60 or dataset.name in computed:
            continue
        try:
            data = Data(dataset.name)
            candidates_and_tsids(data)
        except:
            print(f"Error happened with dataset {dataset.name}")


if __name__ == "__main__":
    run()
