import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def get_windows(ts, window_size):
    windows = sliding_window_view(ts, window_size)
    windows = StandardScaler().fit_transform(windows.T).T
    return windows


def get_most_similar(window, ts):
    windows = get_windows(ts, window.shape[0])
    distances = cdist(windows, window.reshape(1, -1)).squeeze()
    idx = np.argmin(distances)
    return idx, distances[idx]


def mean_distance(window, ts_list):
    sum_dists = 0
    for ts in ts_list:
        _, dist = get_most_similar(window, ts)
        sum_dists += dist
    return sum_dists / len(ts_list)


def silhouette(window, same, different):
    a = mean_distance(window, same)
    b = mean_distance(window, different)
    return (b - a) / max(a, b)
