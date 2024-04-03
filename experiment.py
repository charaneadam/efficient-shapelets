from data import Data
from clustering import Kmeans, ClusterInfo


class Experiment:
    def __init__(self, dataset_name, n_clusters):
        self.dataset = dataset_name
        self.n_clusters = n_clusters
        self.data = Data(dataset_name)
        self.cluster_info = ClusterInfo(self.data, Kmeans(n_clusters))

    def run(self):
        self.cluster_info.init()
        return self.cluster_info.get_info_df()
