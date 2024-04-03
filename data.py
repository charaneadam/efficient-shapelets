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
        self.windows_dists = None
        self.cluster_dists = None

    def get_sliding_windows(self, window_length=25, normalize=True):
        if self.window_size is None or self.window_size != window_length:
            self.window_size = window_length
            self.windows = sliding_window_view(
                self.X_train, window_shape=window_length, axis=1
            ).reshape(-1, window_length)
            if normalize:
                self.windows = StandardScaler().fit_transform(self.windows.T).T
            self.windows_dists = squareform(pdist(self.windows))
        return self.windows

    def assign_clusters_to_windows(self, clusters):
        self.clusters = clusters

    def _cluster_info(self, same_cluster):
        number_of_windows_per_ts = self.ts_length - self.window_size + 1
        unique_ts = set()  # Time series covered
        windows_classes = set()  # Classes of the TS that windows belongs to
        cluster_info = dict()
        for window_id in same_cluster:
            ts_id = window_id // number_of_windows_per_ts
            # window has same class label as the TS containing it
            window_class_label = self.y_train[ts_id]
            # name of the column in Pandas
            label_str = f"label {window_class_label}"
            # keep track of classes in the current cluster
            windows_classes.add(window_class_label)
            # increment the count of label of the window
            cluster_info[label_str] = cluster_info.get(label_str, 0) + 1
            if ts_id not in unique_ts:
                # Keep track of how many different TS are covered
                uniq_lbl = f"# ts from {window_class_label}"
                cluster_info[uniq_lbl] = cluster_info.get(uniq_lbl, 0) + 1
                unique_ts.add(ts_id)
        return cluster_info, windows_classes

    def _add_dominant_class_info(self, windows_classes, cluster_info):
        # Find dominant class and its percentage
        tot_sum = 0
        cur_sum = 0
        best_sum = 0
        best_lbl = None
        for class_label in windows_classes:
            cur_sum = cluster_info[f"label {class_label}"]
            tot_sum += cur_sum
            if best_sum < cur_sum:
                best_sum = cur_sum
                best_lbl = class_label
        cluster_info["dominant label"] = best_lbl
        cluster_info["%"] = 100 * best_sum / tot_sum

    def _info_to_df(self, res):
        df = pd.DataFrame.from_dict(res).T.sort_values("%", ascending=False)
        df.fillna(0, inplace=True)
        for col in df.columns:
            if col in {"Silhouette", "%"}:
                continue
            df[col] = df[col].astype(np.int16)
        return df

    def get_clusters_stats(self):
        cluster_ids = np.unique(self.clusters)
        res = dict()

        for cluster_id in cluster_ids:
            same_cluster = np.where(self.clusters == cluster_id)[0]

            cluster_info, windows_classes = self._cluster_info(same_cluster)

            self._add_dominant_class_info(windows_classes, cluster_info)

            res[cluster_id] = cluster_info

        return self._info_to_df(res)
