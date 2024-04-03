from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from aeon.datasets import load_from_tsfile
import numpy as np
import pandas as pd


def get_dataset(dataset_name, train, path) -> tuple[np.ndarray, np.ndarray]:
    split = "TRAIN" if train else "TEST"
    filepath = f"{path}/{dataset_name}/{dataset_name}_{split}.ts"
    x, y = load_from_tsfile(filepath)
    n_ts, _, ts_length = x.shape
    x = x.reshape((n_ts, ts_length))
    y = y.astype(int)
    return x, y


class Data:
    def __init__(self, dataset_name, path="data"):
        x, y = get_dataset(dataset_name=dataset_name, train=True, path=path)
        self.x_train: np.ndarray = x
        self.y_train: np.ndarray = y
        self.n_ts = self.x_train.shape[0]
        self.windows = None
        self.window_size: int = 0
        self.clusters: np.ndarray

    def get_sliding_windows(self, window_length=25, normalize=True):
        if self.window_size != window_length:
            self.window_size = window_length
            self.windows = sliding_window_view(
                self.x_train, window_shape=window_length, axis=1
            ).reshape(-1, window_length)
            if normalize:
                self.windows = StandardScaler().fit_transform(self.windows.T).T
        return self.windows

    def assign_clusters_to_windows(self, clusters):
        self.clusters = clusters


class Data_info:
    def __init__(self, data) -> None:
        self.data: Data = data
        self.info_df: pd.DataFrame = None

    def _cluster_info(self, same_cluster):
        ts_length = self.data.x_train.shape[1]
        number_of_windows_per_ts = ts_length - self.data.window_size + 1
        unique_ts = set()  # Time series covered
        windows_classes = set()  # Classes of the TS that windows belongs to
        cluster_info = {}
        for window_id in same_cluster:
            ts_id = window_id // number_of_windows_per_ts
            # window has same class label as the TS containing it
            window_class_label = self.data.y_train[ts_id]
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
        self.info_df = df

    def _generate_clusters_info(self):
        cluster_ids = np.unique(self.data.clusters)
        res = {}
        for cluster_id in cluster_ids:
            same_cluster = np.where(self.data.clusters == cluster_id)[0]
            cluster_info, windows_classes = self._cluster_info(same_cluster)
            self._add_dominant_class_info(windows_classes, cluster_info)
            res[cluster_id] = cluster_info
        self._info_to_df(res)

    def get_clusters_labels(self):
        if self.info_df is None:
            self._generate_clusters_info()
        res = list(zip(self.info_df.index, self.info_df["dominant label"]))
        return np.array(sorted(res))[:, 1]

    def get_info_df(self):
        if self.info_df is None:
            self._generate_clusters_info()
        return self.info_df
