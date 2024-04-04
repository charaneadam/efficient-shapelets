import numpy as np
import pandas as pd

from data import Data
from .algorithms import ClusterAlgorithm


class ClusterInfo:
    def __init__(self, cluster_id, data, windows_indices) -> None:
        self.cluster_id: int = cluster_id
        self.w_indices = windows_indices

        self.cluster_size: int = len(self.w_indices)
        self.label: int
        self.popularity: float
        self.ts_covered: set

        self._set_popularity(data)

    def _set_popularity(self, data):
        labels, covered = data.windows_labels_and_covered_ts(self.w_indices)
        labels, counts = np.unique(labels, return_counts=True)
        count, label = max(zip(counts, labels))
        self.label = label
        self.popularity = count / self.cluster_size
        self.ts_covered = covered[label]

    def get_info_for_dataframe(self, data):
        return [
            self.cluster_id,
            self.cluster_size,
            self.label,
            self.popularity * 100,
            len(self.ts_covered),
            sum(data.y_train == self.label),
        ]


class ClustersInfo:
    def __init__(self, data, algorithm) -> None:
        self.data: Data = data
        self.algorithm: ClusterAlgorithm = algorithm
        self.clusters_info: dict = {}
        self.info_df: pd.DataFrame
        self._init()

    def _init(self):
        self.algorithm.run(self.data.get_sliding_windows())
        self._generate_clusters_info()
        self._generate_info_dataframe()

    def _generate_clusters_info(self):
        windows_clusters = self.algorithm.assigned_clusters()
        clusters_ids = np.unique(windows_clusters)
        for cluster_id in clusters_ids:
            same_cluster = np.where(windows_clusters == cluster_id)[0]
            cluster_info = ClusterInfo(cluster_id, self.data, same_cluster)
            self.clusters_info[cluster_id] = cluster_info

    def _generate_info_dataframe(self):
        result = []
        for cinfo in self.clusters_info.values():
            result.append(cinfo.get_info_for_dataframe(self.data))

        cols = ["id", "size", "label", "popularity", "covered", "total"]
        df = pd.DataFrame(result, columns=cols)
        self.info_df = df.sort_values(by="popularity", ascending=False)

    def info(self):
        return self.info_df

    def get_class_labels(self):
        return np.unique(self.data.y_train)

    def get_clusters_of_labels(self, label, top=3):
        return self.info_df[self.info_df.label == label].id.values[:top]

    def best_windows_to_cluster(self, cluster_id, top=3):
        indices = self.clusters_info[cluster_id].w_indices
        label = self.clusters_info[cluster_id].label
        dists = self.algorithm.windows_dists_to_cluster(indices, cluster_id)
        sorted_dists = np.array(sorted(list(zip(dists, indices))))[:, 1]
        selected_windows = []
        count = 0
        for index in sorted_dists.astype(int):
            if self.data.get_window_label(index) == label:
                count += 1
                selected_windows.append(index)
            if count == top:
                break
        return np.array(selected_windows)
