from time import perf_counter
import numpy as np
from numba import njit, prange

from src.storage.data import Data, Windows
from .db import save, init_windows_tables
from src.benchmarks.windows_evaluation.utils import distance_numba


@njit(cache=True, fastmath=True)
def _eval_silhouette_bruteforce(X, y, windows, windows_per_ts):
    n_windows = windows.shape[0]
    n_ts = X.shape[0]
    res = np.zeros(n_windows)
    for window_id in prange(n_windows):
        window = windows[window_id]
        window_ts_id = window_id // windows_per_ts
        window_label = y[window_ts_id]
        a, b = 0.0, 0.0
        a_tot = 0
        b_tot = 0
        for ts_id in prange(n_ts):
            if window_ts_id == ts_id:
                continue
            dist = distance_numba(X[ts_id], window)
            if y[ts_id] == window_label:
                a += dist
                a_tot += 1
            else:
                b += dist
                b_tot += 1
        a /= a_tot
        b /= b_tot
        if b > a:
            mx = b
        else:
            mx = a
        res[window_id] = (b - a) / mx
    return res


def bruteforce(data: Data, window_manager: Windows):
    windows = window_manager.get_windows(data.X_train)
    start = perf_counter()
    results = _eval_silhouette_bruteforce(
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
    from src.storage.database import db

    init_windows_tables(db)

    dataset_name = "CBF"
    data = Data(dataset_name)
    windows_size = 40
    window_skip = int(0.1 * windows_size)
    window_manager = Windows(windows_size, window_skip)
    bruteforce(data, window_manager)
