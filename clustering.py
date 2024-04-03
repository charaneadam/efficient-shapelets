import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


class Cluster:
    def __init__(self, n_centroids: int) -> None:
        self.n_centroids: int = n_centroids
        self.algorithm = KMeans(n_clusters=self.n_centroids)
        self.distances: np.ndarray
        self.centroids_dists: np.ndarray
        self.centroids_labels: np.ndarray

    def run(self, data):
        self.distances = self.algorithm.fit_transform(data)
        self.centroids_dists = pdist(self.algorithm.cluster_centers_)
        self.centroids_dists = squareform(self.centroids_dists)

    def set_clusters_labels(self, labels):
        self.centroids_labels = labels

    def get_elements_labels(self):
        return self.algorithm.labels_

    def get_distances_to_centroids(self):
        return self.distances
