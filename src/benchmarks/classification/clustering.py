from multiprocessing import Pool
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit, objmode, prange
import faiss

from src.benchmarks.classification.utils import _classify, transform
from src.benchmarks.windows_evaluation.utils import (
    distance_numba,
    fstat,
    info_gain,
    silhouette,
)
from src.classifiers import CLASSIFIERS_NAMES
from src.storage.data import Data, Windows
from src.storage.database import SAME_LENGTH_CLASSIFICATION_TABLE_NAME

from src.storage.database import engine


def get_ts_ids():
    query = f"""SELECT DISTINCT id
    FROM (SELECT id, (train+test)*length as size
            FROM dataset INNER JOIN {SAME_LENGTH_CLASSIFICATION_TABLE_NAME}
            ON id=dataset_id ORDER BY size ASC);"""
    return pd.read_sql(query, engine).values.squeeze()


@njit(fastmath=True, parallel=True)
def _eval_bruteforce(X, y, candidates, centroids_labels):
    n_windows = len(candidates)
    n_ts = X.shape[0]
    res = np.zeros((n_windows, 6))  # 6: 3 for sil,infogain,fstat, and 3 for time
    for window_id in prange(n_windows):
        window = candidates[window_id]
        window_label = centroids_labels[window_id]
        dists_to_ts = np.zeros(n_ts)
        for ts_id in range(n_ts):
            dist = distance_numba(X[ts_id], window)
            dists_to_ts[ts_id] = dist

        with objmode(start="f8"):
            start = perf_counter()
        silhouette_score = silhouette(dists_to_ts, window_label, y)
        with objmode(end="f8"):
            end = perf_counter()
        silhouette_time = end - start
        res[window_id][0] = silhouette_score
        res[window_id][1] = silhouette_time

        with objmode(start="f8"):
            start = perf_counter()
        fstat_score = fstat(dists_to_ts, window_label, y)
        with objmode(end="f8"):
            end = perf_counter()
        fstat_time = end - start
        res[window_id][2] = fstat_score
        res[window_id][3] = fstat_time

        with objmode(start="f8"):
            start = perf_counter()
        infgain_score = info_gain(dists_to_ts, window_label, y)
        with objmode(end="f8"):
            end = perf_counter()
        infogain_time = end - start
        res[window_id][4] = infgain_score
        res[window_id][5] = infogain_time
    return res


def assign_labels_to_centroids(data, wm, indices, n_centroids):
    clusters = [[] for _ in range(n_centroids)]
    for subsequence_index, centroid_index in enumerate(indices.squeeze()):
        window_ts_index = wm.get_ts_index_of_window(subsequence_index)
        window_label = data.y_train[window_ts_index]
        clusters[centroid_index].append(window_label)
    labels, counts = np.unique(clusters[0], return_counts=True)
    assert len(clusters[0]) == sum(counts)
    centroids_info = []
    for centroid_id in range(n_centroids):
        labels, counts = np.unique(clusters[centroid_id], return_counts=True)
        num_points_assigned = len(clusters[centroid_id])
        assert num_points_assigned == sum(counts)
        popularity = 0
        assigned_label = 0
        for label, cnt in zip(labels, counts):
            pop = cnt / num_points_assigned
            if pop > assigned_label:
                popularity = pop
                assigned_label = label
        centroids_info.append([assigned_label, popularity, num_points_assigned])
    cols = ["label", "popularity", "population size"]
    df = pd.DataFrame(centroids_info, columns=cols).sort_values(by=cols[-1])
    return df


def select_best_k(df, centroids, method, label, K):
    indices = df[df.label == label].sort_values(by=method).index[:K]
    return [centroids[idx] for idx in indices]


def classify(df, data, centroids, method, K):
    labels = set(data.y_train)
    shapelets = []
    for label in labels:
        shapelets.extend(select_best_k(df, centroids, method, label, K))
    X_tr, X_te = transform(data, shapelets)
    classification_results = {}
    with Pool(len(CLASSIFIERS_NAMES)) as p:
        results = [
            p.apply_async(_classify, (clf_name, X_tr, data.y_train, X_te, data.y_test))
            for clf_name in CLASSIFIERS_NAMES
        ]
        p.close()
        p.join()
    for res, clf_name in zip(results, CLASSIFIERS_NAMES):
        fit_time, predict_time, acc, f1, labels, precision, recall = res.get()
        classification_results[clf_name] = [acc, f1, fit_time, predict_time]
    return classification_results


def compare(data, window_length, method, K):
    wm = Windows(window_length)
    windows = wm.get_windows(data.X_train)
    n_centroids = windows.shape[0] // 20
    km = faiss.Kmeans(window_length, n_centroids)
    km.train(windows)
    dists, indices = km.index.search(windows, 1)
    df = assign_labels_to_centroids(data, wm, indices, n_centroids)
    scores_info = _eval_bruteforce(
        data.X_train, data.y_train, km.centroids, df.label.values
    )
    scores = pd.DataFrame(
        scores_info,
        columns=[
            "silhouette",
            "silhouette time",
            "fstat",
            "fstat time",
            "gain",
            "gain time",
        ],
    )
    evaluation_df = pd.concat([df, scores], axis=1)
    evaluation_df["window size"] = window_length
    classif_res = classify(evaluation_df, data, km.centroids, method, K)
    models_results = [
        classif_res[clf_name] + [method, K, window_length] + [clf_name]
        for clf_name in CLASSIFIERS_NAMES
    ]
    cols = [
        "accuracy",
        "f1",
        "fit time",
        "test time",
        "method",
        "top_K",
        "window_size",
        "classifier",
    ]
    classif_results = pd.DataFrame(models_results, columns=cols)
    return evaluation_df, classif_results


def cluster_dataset(dataset_id):
    dataset_name = str(
        pd.read_sql(
            f"SELECT name FROM dataset WHERE id={dataset_id}", engine
        ).values.squeeze()
    )
    data = Data(dataset_name)
    lengths = pd.read_sql(
        f"""SELECT DISTINCT window_size
        FROM {SAME_LENGTH_CLASSIFICATION_TABLE_NAME}
        WHERE dataset_id={dataset_id}""",
        engine,
    ).values.squeeze()
    for length in lengths:
        for method in ["silhouette", "gain", "fstat"]:
            for K in [3, 5, 10, 20, 50, 100]:
                eval_df, classif_df = compare(data, length, method, K)
                eval_df["dataset_id"] = dataset_id
                eval_df.to_sql("centroids_evaluation", engine, if_exists="append")
                classif_df["dataset_id"] = dataset_id
                classif_df.to_sql(
                    "centroids_classification", engine, if_exists="append"
                )


def run():
    dataset_ids = get_ts_ids()
    for dataset_id in dataset_ids:
        cluster_dataset(dataset_id)


if __name__ == "__main__":
    set_num_threads(NUM_THREADS)
    run()
