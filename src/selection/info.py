import numpy as np
import pandas as pd


class ClusterInfo:
    def __init__(self, cluster_id, data, window_indices) -> None:
        self.cluster_id: int = cluster_id

        self.cluster_size: int = len(window_indices)
        self.popular_label: int
        self.popularity: float
        self.ts_covered: set

        self._set_popularity(data, window_indices)

    def _set_popularity(self, data, windows_indices):
        labels, covered = data.windows_labels_and_covered_ts(windows_indices)
        labels, counts = np.unique(labels, return_counts=True)
        count, label = max(zip(counts, labels))
        self.popular_label = label
        self.popularity = count / self.cluster_size
        self.ts_covered = covered[label]

    def get_info_for_dataframe(self, data):
        return [
            self.cluster_id,
            self.cluster_size,
            self.popular_label,
            self.popularity * 100,
            len(self.ts_covered),
            sum(data.y_train == self.popular_label),
        ]


class ClustersInfo:
    def __init__(self, data, algorithm) -> None:
        self.data = data
        self.algorithm = algorithm
        self.clusters_info: list = []
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
            self.clusters_info.append(cluster_info)

    def _generate_info_dataframe(self):
        result = []
        for cinfo in self.clusters_info:
            result.append(cinfo.get_info_for_dataframe(self.data))

        cols = ["id", "size", "label", "popularity", "covered", "total"]
        df = pd.DataFrame(result, columns=cols)
        self.info_df = df.sort_values(by="popularity", ascending=False)

    def info(self):
        return self.info_df
