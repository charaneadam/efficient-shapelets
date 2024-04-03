from data import Data
from clustering import Cluster, ClusterInfo


class Experiment:
    def __init__(self, dataset_name, n_clusters):
        self.dataset = dataset_name
        self.n_clusters = n_clusters
        self.data = Data(dataset_name)
        self.algorithm = Cluster(n_clusters)
        self.cluster_info = ClusterInfo(self.data)

    def run(self):
        self.algorithm.run(self.data.get_sliding_windows())

        windows_clusters = self.algorithm.assigned_clusters()
        self.cluster_info.assign_clusters_to_windows(windows_clusters)

        clusters_labels = self.cluster_info.get_clusters_labels()
        self.cluster_info.set_clusters_labels(clusters_labels)

        return self.cluster_info.get_info_df()
