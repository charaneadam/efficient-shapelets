import pandas as pd

from src.storage.database import fix_engine, paper_engine
from src.storage.data import Data
from src.benchmarks.windows_evaluation.utils import (
    distance_numba,
    silhouette,
    fstat,
    info_gain,
    info_gain_multiclass,
)


def run_dataset(dataset_id, dataset_name):
    pass

    data = Data(dataset_name)
    query = f"SELECT * FROM candidates WHERE dataset = {dataset_id}"
    df = pd.read_sql(query, fix_engine).drop(columns=["dataset"])

    candidates = dict()
    for label in df.label.unique():
        candidates[label] = []

    evals = []
    for cand_id, _, ts_id, label, start, end in df.values:
        candidate = data.X_train[ts_id][start:end]
        dists = []
        for ts in data.X_train:
            dists.append(distance_numba(ts, candidate))
        sil = silhouette(dists, label, data.y_train, ts_id)
        fstt = fstat(dists, label, data.y_train, ts_id)
        gain1 = info_gain(dists, label, data.y_train)
        gain2 = info_gain_multiclass(dists, label, data.y_train)
        evals.append([cand_id, sil, fstt, gain1, gain2])

    pd.DataFrame(
        evals,
        columns=["candidate", "silhouette", "fstat", "binary info", "multiclass info"],
    ).to_sql("evaluation", fix_engine, if_exists="append", index=False)


if __name__ == "__main__":
    query = "SELECT dataset FROM candidates GROUP BY dataset ORDER BY COUNT(*)"
    datasets = pd.read_sql(query, fix_engine).dataset.values
    datasets_info = pd.read_sql("dataset", paper_engine)
    for dataset_id in datasets[:4]:
        dataset_name = datasets_info[datasets_info.id == dataset_id].name.values[0]
        run_dataset(dataset_id, dataset_name)
