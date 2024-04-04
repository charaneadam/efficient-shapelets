from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.cluster import KMeans


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
