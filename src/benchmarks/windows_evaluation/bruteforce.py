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
def _eval_bruteforce(X, y, windows, window_ts_ids):
    n_windows = len(windows)
    n_ts = X.shape[0]
    res = np.zeros((n_windows, 6))  # 6: 3 for sil,infogain,fstat, and 3 for time
    for window_id in prange(n_windows):
        window = windows[window_id]
        window_ts_id = window_ts_ids[window_id]
        window_label = y[window_ts_id]
        dists_to_ts = np.zeros(n_ts)
        for ts_id in range(n_ts):
            if window_ts_id == ts_id:
                continue
            dist = distance_numba(X[ts_id], window)
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
