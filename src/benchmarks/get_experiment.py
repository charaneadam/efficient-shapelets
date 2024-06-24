import numpy as np
from src.storage.data import Dataset
from src.benchmarks.windows_evaluation.db import (
    WindowsEvaluation,
    WindowsEvaluationAproach,
    ClusteringParametersEvaluation,
    WindowEvaluationClustering,
)


def get_datasets():
    return (
        Dataset.select()
        .where(Dataset.missing_values == False)
        .order_by(Dataset.train * Dataset.length)
    )


def get_approach_id(approach_name):
    return WindowsEvaluationAproach.get(WindowsEvaluationAproach.name == approach_name)


def get_windows_evaluations(dataset, approach):
    return WindowsEvaluation.select().where(
        (WindowsEvaluation.dataset == dataset)
        & (WindowsEvaluation.approach == approach)
    )


def get_missing_window_size(dataset, approach_id):
    windows = get_windows_evaluations(dataset, approach_id)
    res = []
    sizes_covered = {window.window_size for window in windows}
    for window_percentage in np.arange(0.1, 0.61, 0.1):
        window_size = int(dataset.length * window_percentage)
        if window_size in sizes_covered:
            continue
        res.append(window_size)
    return res


def get_centroids_evaluations(dataset, approach):
    return WindowsEvaluation.select().where(
        (WindowsEvaluation.dataset == dataset)
        & (WindowsEvaluation.approach == approach)
    )


def get_missing_centroids(dataset):
    ts_length = dataset.length
    n_ts = dataset.train
    missing = dict()
    for window_percentage in np.arange(0.1, 0.61, 0.1):
        window_size = int(dataset.length * window_percentage)
        evaluations = ClusteringParametersEvaluation.select().where(
            (ClusteringParametersEvaluation.dataset == dataset)
            & (ClusteringParametersEvaluation.window_size == window_size)
        )
        covered_centroids = {ev.n_centroids for ev in evaluations}
        n_windows = int(n_ts * 0.9 * (ts_length - window_size + 1))
        min_centroids = n_windows // 50
        max_centroids = n_windows // 5
        increase = (max_centroids - min_centroids) // 10
        for n_centroids in np.arange(min_centroids, max_centroids, increase):
            if n_centroids in covered_centroids:
                continue
            missing[window_size] = missing.get(window_size, [])
            missing[window_size].append(n_centroids)
    return missing
