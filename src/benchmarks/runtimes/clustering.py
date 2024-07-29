import pandas as pd
from time import perf_counter
import faiss
from numba import set_num_threads


from src.config import NUM_THREADS
from src.exceptions import DataFailure
from src.storage.database import paper_engine
from src.storage.data import Data, Windows
from src.benchmarks.classification.clustering_variable_length import (
    compute_distances_shapelets_ts,
)


def run():
    query = "SELECT id, name, train, n_classes, length, length*train AS n_timestamps FROM dataset ORDER BY length*train"
    datasets = pd.read_sql(query, paper_engine)
    datasets.head()

    results = []
    for dataset_name in datasets.name.values:
        try:
            data = Data(dataset_name)
        except DataFailure:
            print(f"Missing values in dataset {dataset_name}")
            continue
        ts_length = data.X_train.shape[1]
        lengths = list(
            map(lambda x: int(x * ts_length), [0.05, 0.1, 0.2, 0.3, 0.5, 0.6])
        )
        results = []
        try:
            for window_length in lengths:
                wm = Windows(window_length)
                windows = wm.get_windows(data.X_train)

                start = perf_counter()
                n_centroids = windows.shape[0] // 30
                km = faiss.Kmeans(window_length, n_centroids, verbose=False, niter=5)
                km.train(windows)
                km.index.search(windows, 1)
                end = perf_counter()
                extraction_time = end - start

                start = perf_counter()
                _ = compute_distances_shapelets_ts(data.X_train, km.centroids)
                end = perf_counter()
                distances_time = end - start

                result = [
                    dataset_name,
                    window_length,
                    n_centroids,
                    extraction_time,
                    distances_time,
                ]
                results.append(result)
            with open("centroid_runtime.csv", "a") as f:
                for result in results:
                    f.write(",".join(map(str, result)) + "\n")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    set_num_threads(NUM_THREADS)
    faiss.omp_set_num_threads(NUM_THREADS)
    run()
