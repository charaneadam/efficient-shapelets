from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from data import Data


class ClusterAlgorithm(ABC):
    @abstractmethod
    def run(self, data):
        raise NotImplementedError("Run your algorithm and save info needed")

    @abstractmethod
    def assigned_clusters(self):
        raise NotImplementedError("Return clusters assigned to windows")

    @abstractmethod
    def get_distances_to_references(self):
        raise NotImplementedError("Distance of windows to cluster reference")


class Kmeans(ClusterAlgorithm):
    def __init__(self, n_centroids: int, random_state=0) -> None:
        self.algorithm = KMeans(n_clusters=n_centroids, random_state=random_state)
        self.distances: np.ndarray
        self.centroids_dists: np.ndarray

    def run(self, data):
        self.distances = self.algorithm.fit_transform(data)
        self.centroids_dists = pdist(self.algorithm.cluster_centers_)
        self.centroids_dists = squareform(self.centroids_dists)

    def assigned_clusters(self):
        return self.algorithm.labels_

    def get_distances_to_references(self):
        return self.distances


class ClusterInfo:
    def __init__(self, cluster_id, data, same_cluster) -> None:
        self.cluster_id: int = cluster_id
        self.unique_ts = set()  # Time series covered
        self.info = {}
        self.windows_classes = set()  # Classes of the TS that windows belongs to
        self.data = data
        self.same_cluster = same_cluster
        self._set_info()

    def _get_ts_info(self, window_id):
        ts_length = self.data.x_train.shape[1]
        number_of_windows_per_ts = ts_length - self.data.window_size + 1
        ts_id = window_id // number_of_windows_per_ts
        # window has same class label as the TS containing it
        ts_class = self.data.y_train[ts_id]
        # keep track of classes in the current cluster
        self.windows_classes.add(ts_class)
        return ts_id, ts_class

    def _set_ts_and_window_info(self, ts_id, ts_class):
        # Keep track of how many different TS are covered
        self.unique_ts.add(ts_id)

        # name of the column in Pandas
        uniq_lbl = f"# ts from {ts_class}"
        self.info[uniq_lbl] = self.info.get(uniq_lbl, 0) + 1

        # name of the column in Pandas
        label_str = f"label {ts_class}"
        # increment the count of label of the window
        self.info[label_str] = self.info.get(label_str, 0) + 1

    def _set_dominant_class_info(self):
        # Find dominant class and its percentage
        tot_sum = 0
        cur_sum = 0
        best_sum = 0
        best_lbl = None
        for class_label in self.windows_classes:
            cur_sum = self.info[f"label {class_label}"]
            tot_sum += cur_sum
            if best_sum < cur_sum:
                best_sum = cur_sum
                best_lbl = class_label
        self.info["dominant label"] = best_lbl
        self.info["%"] = 100 * best_sum / tot_sum

    def _set_info(self):
        for window_id in self.same_cluster:
            ts_id, ts_class = self._get_ts_info(window_id)
            self._set_ts_and_window_info(ts_id, ts_class)
            self._set_dominant_class_info()


class ClustersInfo:
    def __init__(self, data, algorithm) -> None:
        self.data: Data = data
        self.info_df: pd.DataFrame = None
        self.clusters: np.ndarray
        self.centroids_labels: np.ndarray
        self.algorithm: ClusterAlgorithm = algorithm
        self._init()

    def _set_clusters_labels(self):
        if self.info_df is None:
            self._generate_clusters_info()
        res = list(zip(self.info_df.window_id, self.info_df["dominant label"]))
        self.centroids_labels = np.array(sorted(res))[:, 1]

    def _info_to_df(self, res):
        df = pd.DataFrame.from_dict(res).T.sort_values("%", ascending=False)
        df.fillna(0, inplace=True)
        for col in df.columns:
            if col in {"Silhouette", "%"}:
                continue
            df[col] = df[col].astype(np.int16)
        df.reset_index(inplace=True, names="window_id")
        self.info_df = df

    def _generate_clusters_info(self):
        cluster_ids = np.unique(self.clusters)
        res = {}
        for cluster_id in cluster_ids:
            same_cluster = np.where(self.clusters == cluster_id)[0]
            cluster_info = ClusterInfo(cluster_id, self.data, same_cluster)
            res[cluster_id] = cluster_info.info
        self._info_to_df(res)

    def _init(self):
        self.algorithm.run(self.data.get_sliding_windows())
        windows_clusters = self.algorithm.assigned_clusters()
        self.clusters = windows_clusters
        self._set_clusters_labels()
        self._generate_clusters_info()

    def get_info_df(self):
        return self.info_df
