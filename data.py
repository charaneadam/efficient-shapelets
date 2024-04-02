from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from aeon.datasets import load_from_tsfile
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform


def get_dataset(dataset_name, train, data_path):
    split = "TRAIN" if train else "TEST"
    filepath = f"{data_path}/{dataset_name}/{dataset_name}_{split}.ts"
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name, data_path="data"):
        self.X_train, self.y_train = get_dataset(
            dataset_name, train=True, data_path=data_path
        )
        self.n_ts = self.X_train.shape[0]
        self.ts_length = self.X_train.shape[1]
        self.windows = None
        self.window_size = None
        self.clusters = None
        self.windows_distances = None
        self.cluster_distances = None

    def get_sliding_windows(self, window_length=25, normalize=True):
        if self.window_size is None or self.window_size != window_length:
            self.window_size = window_length
            self.windows = sliding_window_view(
                self.X_train, window_shape=window_length, axis=1
            ).reshape(-1, window_length)
            if normalize:
                self.windows = StandardScaler().fit_transform(self.windows.T).T
            self.windows_distances = squareform(pdist(self.windows))
        return self.windows

    def assign_clusters_to_windows(self, clusters):
        self.clusters = clusters

    def get_clusters_stats(self):
        cluster_ids = np.unique(self.clusters)
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        res = dict()
        for cid in cluster_ids:
            clusters = np.where(self.clusters == cid)[0]
            other_clusters = np.where(self.clusters != cid)[0]
            a = self.windows_distances[clusters, :][:, clusters].mean()
            b = self.windows_distances[clusters, :][:, other_clusters].mean()
            silhouette = (b - a) / max(a, b)
            cluster_labels = {"Silhouette": silhouette}
            seen = set()
            labels = set()
            for c in clusters:
                ts_id = c // number_of_windows_per_ts
                window_label = self.y_train[ts_id]
                label_str = f"label {window_label}"
                labels.add(window_label)
                cluster_labels[label_str] = cluster_labels.get(label_str, 0) + 1
                if ts_id not in seen:
                    uniq_lbl = f"# ts from {window_label}"
                    cluster_labels[uniq_lbl] = cluster_labels.get(uniq_lbl, 0) + 1
                    seen.add(ts_id)
            tot_sum = 0
            cur_sum = 0
            best_sum = 0
            best_lbl = None
            for label in labels:
                cur_sum = cluster_labels[f"label {label}"]
                tot_sum += cur_sum
                if best_sum < cur_sum:
                    best_sum = cur_sum
                    best_lbl = label
            cluster_labels["dominant label"] = best_lbl
            cluster_labels["%"] = 100 * best_sum / tot_sum
            res[cid] = cluster_labels
        df = pd.DataFrame.from_dict(res).T.sort_values("%", ascending=False)
        df.fillna(0, inplace=True)
        for col in df.columns:
            if col == "Silhouette" or col == "%":
                continue
            df[col] = df[col].astype(np.int16)
        return df
