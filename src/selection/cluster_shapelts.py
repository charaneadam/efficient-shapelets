from src.data import Data
from algorithms import Kmeans
from info import ClustersInfo


class ClusteringShapelets:
    def __init__(self, data: Data, n_centers: int = 100, threshold: float = 0.7):
        self.data = data
        self.n_centers = n_centers
        self.threshold = threshold

    def get_shapelets_ids(self):
        alg = Kmeans(self.n_centers)
        cinfo = ClustersInfo(self.data, alg)
        centers = cinfo.algorithm.algorithm.cluster_centers_
    # TODO: Return Windows
