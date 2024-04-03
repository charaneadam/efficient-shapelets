import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from data import Data


class Cluster:
    def __init__(self, n_centroids: int) -> None:
        self.n_centroids: int = n_centroids
        self.algorithm = KMeans(n_clusters=self.n_centroids, random_state=0)
        self.distances: np.ndarray
        self.centroids_dists: np.ndarray

    def run(self, data):
        self.distances = self.algorithm.fit_transform(data)
        self.centroids_dists = pdist(self.algorithm.cluster_centers_)
        self.centroids_dists = squareform(self.centroids_dists)

    def assigned_clusters(self):
        return self.algorithm.labels_

    def get_distances_to_centroids(self):
        return self.distances


class ClusterInfo:
    def __init__(self, data, algorithm) -> None:
        self.data: Data = data
        self.info_df: pd.DataFrame = None
        self.clusters: np.ndarray
        self.centroids_labels: np.ndarray
        self.algorithm: Cluster = algorithm

    def init(self):
        self.algorithm.run(self.data.get_sliding_windows())
        windows_clusters = self.algorithm.assigned_clusters()
        self.clusters = windows_clusters
        self.set_clusters_labels()


    def set_clusters_labels(self):
        if self.info_df is None:
            self._generate_clusters_info()
        res = list(zip(self.info_df.window_id, self.info_df["dominant label"])) 
        self.centroids_labels = np.array(sorted(res))[:, 1]

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
        df.reset_index(inplace=True, names="window_id")
        self.info_df = df

    def _generate_clusters_info(self):
        cluster_ids = np.unique(self.clusters)
        res = {}
        for cluster_id in cluster_ids:
            same_cluster = np.where(self.clusters == cluster_id)[0]
            cluster_info, windows_classes = self._cluster_info(same_cluster)
            self._add_dominant_class_info(windows_classes, cluster_info)
            res[cluster_id] = cluster_info
        self._info_to_df(res)

    def get_info_df(self):
        if self.info_df is None:
            self._generate_clusters_info()
        return self.info_df
