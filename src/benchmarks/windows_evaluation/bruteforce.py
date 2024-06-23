from time import perf_counter
import numpy as np
from numba import njit, prange, objmode

from src.storage.data import Data, Windows
from .db import save
from src.benchmarks.windows_evaluation.utils import (
    distance_numba,
    fstat,
    info_gain,
    silhouette,
)


@njit(cache=True, fastmath=True, parallel=True)
def _eval_bruteforce(X, y, windows, windows_per_ts):
    n_windows = windows.shape[0]
    n_ts = X.shape[0]
    res = np.zeros((n_windows, 6))# 6: 3 for sil,infogain,fstat, and 3 for time
    for window_id in prange(n_windows):
        window = windows[window_id]
        window_ts_id = window_id // windows_per_ts
        window_label = y[window_ts_id]
        dists_to_ts = np.zeros(n_ts)
        for ts_id in range(n_ts):
            if window_ts_id == ts_id:
                continue
            dist = distance_numba(X[ts_id], window)
            dists_to_ts[ts_id] = dist

        with objmode(start='f8'):
            start = perf_counter()
        silhouette_score = silhouette(dists_to_ts, window_label, y, window_ts_id)
        with objmode(end='f8'):
            end = perf_counter()
        silhouette_time = end - start
        res[window_id][0] = silhouette_score
        res[window_id][1] = silhouette_time

        with objmode(start='f8'):
            start = perf_counter()
        fstat_score = fstat(dists_to_ts, window_label, y, window_ts_id)
        with objmode(end='f8'):
            end = perf_counter()
        fstat_time = end - start
        res[window_id][2] = fstat_score
        res[window_id][3] = fstat_time

        with objmode(start='f8'):
            start = perf_counter()
        infgain_score = info_gain(dists_to_ts, window_label, y, window_ts_id)
        with objmode(end='f8'):
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


if __name__ == "__main__":
    dataset_name = "CBF"
    data = Data(dataset_name)
    windows_size = 40
    window_skip = int(0.1 * windows_size)
    window_manager = Windows(windows_size, window_skip)
    bruteforce(data, window_manager)
