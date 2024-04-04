from data import Data
from clustering.info import ClustersInfo
from clustering.algorithms import Kmeans


class Experiment:
    def __init__(self, dataset_name, window_size, n_clusters):
        self.data = Data(dataset_name, window_size)
        self.cluster_info = ClustersInfo(self.data, Kmeans(n_clusters))

    def run(self):
        return self.cluster_info.get_info_df()
