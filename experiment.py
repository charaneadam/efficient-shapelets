from data import Data
from clustering import Cluster


class Experiment:
    def __init__(self, dataset_name, n_clusters):
        self.dataset = dataset_name
        self.n_clusters = n_clusters
        self.data = Data(dataset_name)
        self.algorithm = Cluster(n_clusters)

    def run(self):
        self.algorithm.run(self.data.get_sliding_windows())
        self.data.assign_clusters_to_windows(self.algorithm.get_elements_labels())
        return self.data.get_clusters_stats()




