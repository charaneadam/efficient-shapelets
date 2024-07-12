from numba import set_num_threads
from multiprocessing import Pool
from time import perf_counter
import numpy as np
import pandas as pd
from numba import njit, objmode, prange
import faiss

from src.benchmarks.classification.utils import _classify
from src.benchmarks.windows_evaluation.utils import (
    distance_numba,
    fstat,
    info_gain,
    silhouette,
)
from src.classifiers import CLASSIFIERS_NAMES
from src.config import NUM_THREADS
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
def compute_distances_shapelets_ts(X, candidates):
    n_windows = len(candidates)
    n_ts = X.shape[0]
    distances = np.zeros((n_ts, n_windows))
    for window_id in prange(n_windows):
        window = candidates[window_id]
        for ts_id in range(n_ts):
            dist = distance_numba(X[ts_id], window)
            distances[ts_id, window_id] = dist
    return distances


@njit(fastmath=True)
def compute_silhouette(dists_to_ts, window_label, y):
    with objmode(start="f8"):
        start = perf_counter()
    silhouette_score = silhouette(dists_to_ts, window_label, y)
    with objmode(end="f8"):
        end = perf_counter()
    silhouette_time = end - start
    return silhouette_score, silhouette_time


@njit(fastmath=True)
def compute_fstat(dists_to_ts, window_label, y):
    with objmode(start="f8"):
        start = perf_counter()
    fstat_score = fstat(dists_to_ts, window_label, y)
    with objmode(end="f8"):
        end = perf_counter()
    fstat_time = end - start
    return fstat_score, fstat_time


@njit(fastmath=True)
def compute_gain(dists_to_ts, window_label, y):
    with objmode(start="f8"):
        start = perf_counter()
    infgain_score = info_gain(dists_to_ts, window_label, y)
    with objmode(end="f8"):
        end = perf_counter()
    infogain_time = end - start
    return infgain_score, infogain_time


@njit(fastmath=True, parallel=True)
def _eval_bruteforce(distances, y, candidates, centroids_labels):
    n_candidates = len(candidates)
    # 6: sil,infogain,fstat, and their times
    candidates_evaluations = np.zeros((n_candidates, 6))
    for candidate_id, label in enumerate(centroids_labels):
        dists = distances[:, candidate_id]
        silhouette, time = compute_silhouette(dists, label, y)
        candidates_evaluations[candidate_id, 0] = silhouette
        candidates_evaluations[candidate_id, 1] = time
        fstat, time = compute_fstat(dists, label, y)
        candidates_evaluations[candidate_id, 2] = fstat
        candidates_evaluations[candidate_id, 3] = time
        gain, time = compute_gain(dists, label, y)
        candidates_evaluations[candidate_id, 4] = gain
        candidates_evaluations[candidate_id, 5] = time
    return pd.DataFrame(
        candidates_evaluations,
        columns=[
            "silhouette",
            "silhouette time",
            "fstat",
            "fstat time",
            "gain",
            "gain time",
        ],
    )


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


def select_best_k(df, method, label, K):
    indices = df[df.label == label].sort_values(by=method).index[:K]
    return indices


def get_transformed_data(df, labels, method, train_distances, test_distances, K):
    indices = []
    for label in labels:
        indices.extend(select_best_k(df, method, label, K))
    X_tr = train_distances[:, indices]
    X_te = test_distances[:, indices]
    return X_tr, X_te


def classify(df, data, method, K, train_distances, test_distances):
    X_tr, X_te = get_transformed_data(
        df, set(data.y_train), method, train_distances, test_distances, K
    )
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


def compare(data, evaluation_df, window_length, method, K, distances, test_distances):
    classif_res = classify(evaluation_df, data, method, K, distances, test_distances)
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
    return classif_results


def get_data_and_lengths(dataset_id):
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
    return data, lengths


def get_centroids_and_info(data, length):
    wm = Windows(length)
    windows = wm.get_windows(data.X_train)
    n_centroids = windows.shape[0] // 20
    km = faiss.Kmeans(length, n_centroids)
    km.train(windows)
    dists, indices = km.index.search(windows, 1)
    df = assign_labels_to_centroids(data, wm, indices, n_centroids)
    centroids = km.centroids
    return centroids, df


def get_evaluation_df(data, lengths):
    evaluation_dfs = []
    train_distances = []
    test_distances = []
    for length in lengths:
        centroids, df = get_centroids_and_info(data, length)
        train_dists = compute_distances_shapelets_ts(data.X_train, centroids)
        test_dists = compute_distances_shapelets_ts(data.X_test, centroids)
        train_distances.append(train_dists)
        test_distances.append(test_dists)
        scores = _eval_bruteforce(train_dists, data.y_train, centroids, df.label.values)
        evaluation_df = pd.concat([df, scores], axis=1)
        evaluation_df["window size"] = length
        evaluation_dfs.append(evaluation_dfs)
    return pd.concat(evaluation_dfs), train_distances, test_distances


def cluster_dataset(dataset_id):
    data, lengths = get_data_and_lengths(dataset_id)
    evaluation_df, train_distances, test_distances = get_evaluation_df(data, lengths)
    evaluation_df["dataset_id"] = dataset_id

    classif_dfs = []
    for method in ["silhouette", "gain", "fstat"]:
        for K in [3, 5, 10, 20, 50, 100]:
            classif_df = compare(
                data, evaluation_df, lengths, method, K, train_distances, test_distances
            )
            classif_df["dataset_id"] = dataset_id
            classif_dfs.append(classif_df)
    evaluation_df.to_sql("centroids_evaluation", engine, if_exists="append")
    pd.concat(classif_dfs).to_sql(
        "centroids_classification", engine, if_exists="append"
    )


def run():
    dataset_ids = get_ts_ids()
    for dataset_id in dataset_ids:
        cluster_dataset(dataset_id)


if __name__ == "__main__":
    set_num_threads(NUM_THREADS)
    faiss.omp_set_num_threads(NUM_THREADS)
    run()
