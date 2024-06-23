import numpy as np
from src.storage.data import Dataset
from src.benchmarks.windows_evaluation.db import (
    WindowsEvaluation,
    WindowsEvaluationAproach,
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
