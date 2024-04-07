from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.cluster import KMeans


class ClusterAlgorithm(ABC):
    """
    An abstract class to be used as a base case for clustering algorithms.
    Concrete classes have to implement:
    - run: A method to run the clustering and store all necessary data
    - assigned_clusters: Returns the points assigned to each cluster.
    - distances_between_clusters: Returns distances between clusters
    - windows_dists_to_cluster: return distances between a list of windows
    and a cluster
    """

    @abstractmethod
    def run(self, data):
        raise NotImplementedError("Run your algorithm and save info needed")

    @abstractmethod
    def assigned_clusters(self):
        raise NotImplementedError("Return clusters assigned to windows")

    @abstractmethod
    def distances_between_clusters(self):
        raise NotImplementedError("Distance between clusters")

    @abstractmethod
    def windows_dists_to_cluster(self, windows_indices, cluster_id):
        raise NotImplementedError("Returns distances of windows to the cluster")


class Kmeans(ClusterAlgorithm):
    def __init__(self, n_centroids: int, random_state=0) -> None:
        self.algorithm = KMeans(
            n_clusters=n_centroids, random_state=random_state, n_init="auto"
        )
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

    def distances_between_clusters(self):
        return self.centroids_dists

    def windows_dists_to_cluster(self, windows_indices, cluster_id):
        return self.distances[:, cluster_id][windows_indices]
