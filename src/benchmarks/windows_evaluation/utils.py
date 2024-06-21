import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def distance_numba(ts: np.ndarray, shapelet: np.ndarray):
    n = ts.shape[0]
    length = shapelet.shape[0]
    min_dist = np.inf
    for i in prange(n - length + 1):
        distance = 0.0
        x = ts[i : i + length]
        for j in range(length):
            difference = x[j] - shapelet[j]
            distance += difference * difference
            if distance >= min_dist:
                break
        min_dist = min(min_dist, distance)
    return np.sqrt(min_dist)
