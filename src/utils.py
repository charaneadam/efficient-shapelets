from time import perf_counter
import numpy as np
from numba import njit, objmode

EPS = 0.0000001


@njit(fastmath=True)
def distance_numba(ts: np.ndarray, shapelet: np.ndarray):
    n = ts.shape[0]
    length = shapelet.shape[0]
    min_dist = np.inf
    for i in range(n - length + 1):
        distance = 0.0
        x = ts[i: i + length]
        mu = np.mean(x)
        std = np.std(x)
        for j in range(length):
            difference = ((x[j] - mu) / max(std, EPS)) - shapelet[j]
            distance += difference * difference
            if distance >= min_dist:
                break
        min_dist = min(min_dist, distance)
    return np.sqrt(min_dist)


@njit
def candidate_distances(X, candidate, ts_id):
    candidate = (candidate - np.mean(candidate)) / max(np.std(candidate), EPS)
    m = len(X)
    distances = np.zeros(m)
    for idx in range(m):
        if idx == ts_id:
            continue
        distances[idx] = distance_numba(X[idx], candidate)
    return distances


@njit(fastmath=True)
def silhouette(dists_to_ts, y, ts_idx):
    a, b = 0.0, 0.0
    cnta, cntb = 0, 0
    n_ts = len(dists_to_ts)
    for ts_id in range(n_ts):
        if ts_id == ts_idx:
            continue
        if y[ts_id] == y[ts_idx]:
            a += dists_to_ts[ts_id]
            cnta += 1
        else:
            b += dists_to_ts[ts_id]
            cntb += 1
    a /= cnta
    b /= cntb
    mx = a if a > b else b
    if mx == 0:
        mx = 1
    return (b - a) / mx


@njit(fastmath=True)
def fstat(dists_to_ts, y, ts_idx=-1):
    labels = {label: idx for idx, label in enumerate(np.sort(list(set(y))))}
    n_labels = len(labels)
    n_ts = len(dists_to_ts)

    mean_dists_to_labels = np.zeros(n_labels)
    count_ts_per_label = np.zeros(n_labels)
    for ts_id in range(n_ts):
        if ts_id == ts_idx:
            continue
        ts_label_idx = labels[y[ts_id]]
        mean_dists_to_labels[ts_label_idx] += dists_to_ts[ts_id]
        count_ts_per_label[ts_label_idx] += 1

    global_mean = 0.0
    for label_idx in range(n_labels):
        if count_ts_per_label[label_idx]:
            mean_dists_to_labels[label_idx] /= count_ts_per_label[label_idx]
            global_mean += mean_dists_to_labels[label_idx]

    between_group_variability = 0.0
    for label_idx in range(n_labels):
        diff = mean_dists_to_labels[label_idx] - global_mean
        between_group_variability += diff * diff
    between_group_variability /= n_labels - 1

    within_group_variability = 0.0
    for ts_id in range(n_ts):
        if ts_idx == ts_id:
            continue
        ts_label_idx = labels[y[ts_id]]
        diff = dists_to_ts[ts_id] - mean_dists_to_labels[ts_label_idx]
        within_group_variability += diff * diff
    within_group_variability /= n_ts - n_labels

    within_group_variability = max(within_group_variability, EPS)
    return between_group_variability / within_group_variability


@njit(fastmath=True)
def info_gain(dists_to_ts, label, y, ts_idx=-1):
    n_ts = len(dists_to_ts)
    cnt_same = 0
    cnt_other = 0
    for ts_idx in range(n_ts):
        if label == y[ts_idx]:
            cnt_same += 1
        else:
            cnt_other += 1
    p = cnt_same / n_ts
    gain_before_split = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    dists_and_ts_indices = [(dist, idx)
                            for idx, dist in enumerate(dists_to_ts)]
    sorted_dists = sorted(dists_and_ts_indices)
    same = 0
    other = 0
    quality = 0
    for split_point in range(1, n_ts - 1):
        _, idx = sorted_dists[split_point - 1]
        if y[idx] == label:
            same += 1
        else:
            other += 1
        p_left = same / split_point
        gain_left = 0
        if p_left:
            gain_left = -p_left * np.log2(p_left)
        if p_left < 1:
            gain_left -= (1 - p_left) * np.log2(1 - p_left)
        p_right = (cnt_same - same) / (n_ts - split_point)
        gain_right = 0
        if gain_right:
            gain_right = -p_right * np.log2(p_right)
        if gain_right < 1:
            gain_right -= (1 - p_right) * np.log2(1 - p_right)
        gain = (split_point / n_ts) * gain_left
        gain += ((n_ts - split_point) / n_ts) * gain_right
        info = gain_before_split - gain
        quality = max(info, quality)
    return quality


@njit(fastmath=True)
def info_gain_multiclass(dists_to_ts, y, ts_idx=-1):
    labels = np.unique(y)
    n_ts = len(dists_to_ts)

    count_labels_left = {label: 0 for label in labels}
    count_labels_right = {label: 0 for label in labels}
    for label in y:
        count_labels_right[label] += 1

    gain_before_split = 0
    for count in count_labels_right.values():
        p = count / n_ts
        gain_before_split -= p * np.log2(p)

    dists_ts_indices = [(dist, idx) for idx, dist in enumerate(dists_to_ts)]
    sorted_dists = sorted(dists_ts_indices)

    quality = 0
    for split_point in range(1, n_ts - 1):
        _, idx = sorted_dists[split_point - 1]
        count_labels_left[y[idx]] += 1
        count_labels_right[y[idx]] -= 1

        entropy_left, entropy_right = 0, 0
        for count in count_labels_left.values():
            if count == 0:
                continue
            p = count / split_point
            entropy_left -= p * np.log2(p)
        for count in count_labels_right.values():
            if count == 0:
                continue
            p = count / (n_ts - split_point)
            entropy_right -= p * np.log2(p)
        gain = gain_before_split
        gain -= (split_point / n_ts) * entropy_left
        gain += ((n_ts - split_point) / n_ts) * entropy_right
        quality = max(quality, gain)
    return quality


@njit
def compute_silhouette(distances, y, ts_id):
    with objmode(start="f8"):
        start = perf_counter()
    score = silhouette(distances, y, ts_id)
    with objmode(end="f8"):
        end = perf_counter()
    return score, end-start


@njit
def compute_fstat(distances, y, ts_id):
    with objmode(start="f8"):
        start = perf_counter()
    score = fstat(distances, y, ts_id)
    with objmode(end="f8"):
        end = perf_counter()
    return score, end-start


@njit
def compute_gain(distances, y, ts_id):
    with objmode(start="f8"):
        start = perf_counter()
    score = info_gain_multiclass(distances, y, ts_id)
    with objmode(end="f8"):
        end = perf_counter()
    return score, end-start


@njit
def compute_distances(X, candidate, ts_id):
    with objmode(start="f8"):
        start = perf_counter()
    distances = candidate_distances(X, candidate, ts_id)
    with objmode(end="f8"):
        end = perf_counter()
    return distances, end-start


@njit
def evaluate_candidate(X, y, candidate_info):
    candidate_id, ts_id, start, end = candidate_info
    candidate = X[ts_id][start: end]

    distances, distances_time = compute_distances(X, candidate, ts_id)
    silhouette_score, silhouette_time = compute_silhouette(distances, y, ts_id)
    fstat_score, fstat_time = compute_fstat(distances, y, ts_id)
    gain_score, gain_time = compute_gain(distances, y, ts_id)
    return (
        candidate_id,
        distances_time,
        fstat_score,
        silhouette_score,
        gain_score,
        fstat_time,
        silhouette_time,
        gain_time
    )
