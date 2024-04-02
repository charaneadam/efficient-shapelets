from sklearn.cluster import KMeans


class Cluster:
    def __init__(self, n_centroids) -> None:
        self.n_centroids = n_centroids
        self.algorithm = KMeans(n_clusters=self.n_centroids)
        self.distances = None

    def run(self, data):
        self.distances = self.algorithm.fit_transform(data)

    def get_elements_labels(self):
        return self.algorithm.labels_

    def get_distances_to_centroids(self):
        return self.distances
