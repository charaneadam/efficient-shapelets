from data import Data, Data_info
from clustering import Cluster


class Experiment:
    def __init__(self, dataset_name, n_clusters):
        self.dataset = dataset_name
        self.n_clusters = n_clusters
        self.data = Data(dataset_name)
        self.algorithm = Cluster(n_clusters)
        self.data_info = Data_info(self.data)

    def run(self):
        self.algorithm.run(self.data.get_sliding_windows())

        windows_clusters = self.algorithm.get_elements_labels()
        self.data.assign_clusters_to_windows(windows_clusters)

        clusters_labels = self.data_info.get_clusters_labels()
        self.algorithm.set_clusters_labels(clusters_labels)

        return self.data_info.get_info_df()
