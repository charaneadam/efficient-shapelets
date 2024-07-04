import pandas as pd
from time import perf_counter
import numpy as np
from numba import njit, prange, objmode

from src.benchmarks.get_experiment import (
    get_approach_id,
    get_datasets,
    get_missing_window_size,
)
from src.storage.data import Data, Windows
from .db import save
from src.benchmarks.windows_evaluation.utils import (
    distance_numba,
    fstat,
    info_gain,
    silhouette,
)


@njit(fastmath=True, parallel=True)
def _eval_bruteforce(X, y, candidates, window_ts_ids, normalize_by_length=False):
    """Computes the silhouette, fstat and infogain for every candidate as well
    as the timing for each operation

    Parameters
    ----------
    X : array_like of shape (n,m) where n is the number of the time series
        and m is the length. All time series are assumed to be of same length.
    y : array_like of shape(n,). Contains the labels of the time series X.
    candidates : list of arrays. Every array is a candidate.
    window_ts_ids: the id of the time series from which the candidate has been
        extracted from.


    Returns
    -------
    array of shape (K, 6) where K is the number of candidates.
        Indices 0, 2 and 4 are the silhouette, fstat and infogain scores
        respectively. Indices 1,3 and 5 are their corresponding timings.

    """
    n_windows = len(candidates)
    n_ts = X.shape[0]
    res = np.zeros((n_windows, 6))  # 6: 3 for sil,infogain,fstat, and 3 for time
    for window_id in prange(n_windows):
        window = candidates[window_id]
        window_ts_id = window_ts_ids[window_id]
        window_label = y[window_ts_id]
        dists_to_ts = np.zeros(n_ts)
        for ts_id in range(n_ts):
            if window_ts_id == ts_id:
                continue
            dist = distance_numba(X[ts_id], window)
            if normalize_by_length:
                dist /= len(window)  # Should we use sqrt of n?
            dists_to_ts[ts_id] = dist

        with objmode(start="f8"):
            start = perf_counter()
        silhouette_score = silhouette(dists_to_ts, window_label, y, window_ts_id)
        with objmode(end="f8"):
            end = perf_counter()
        silhouette_time = end - start
        res[window_id][0] = silhouette_score
        res[window_id][1] = silhouette_time

        with objmode(start="f8"):
            start = perf_counter()
        fstat_score = fstat(dists_to_ts, window_label, y, window_ts_id)
        with objmode(end="f8"):
            end = perf_counter()
        fstat_time = end - start
        res[window_id][2] = fstat_score
        res[window_id][3] = fstat_time

        with objmode(start="f8"):
            start = perf_counter()
        infgain_score = info_gain(dists_to_ts, window_label, y, window_ts_id)
        with objmode(end="f8"):
            end = perf_counter()
        infogain_time = end - start
        res[window_id][4] = infgain_score
        res[window_id][5] = infogain_time
    return res


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


def bruteforce(data: Data, window_manager: Windows):
    windows = window_manager.get_windows(data.X_train)
    start = perf_counter()
    results = _eval_bruteforce(
        data.X_train, data.y_train, windows, window_manager.windows_per_ts
    )
    end = perf_counter()
    save(
        data.dataset_name,
        "Bruteforce",
        window_manager.size,
        window_manager.skip,
        end - start,
        results,
    )


def run_bruteforce():
    datasets = get_datasets()
    approach_id = get_approach_id("Bruteforce")
    for dataset in datasets:
        missing_sizes = get_missing_window_size(dataset, approach_id)
        if len(missing_sizes) == 0:
            continue
        print(dataset, end=": ")
        data = Data(dataset.name)
        for window_size in missing_sizes:
            print(window_size, end=", ")
            window_skip = int(0.1 * window_size)
            window_manager = Windows(window_size, window_skip)
            try:
                bruteforce(data, window_manager)
            except:
                print("error", end=", ")
        print()


if __name__ == "__main__":
    run_bruteforce()
