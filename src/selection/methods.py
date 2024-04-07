from abc import ABC, abstractmethod
from .info import ClustersInfo


class SelectionMethod(ABC):
    """
    An abstract class that contains one method (select) that
    that returns selected shapelets.
    """
    @abstractmethod
    def select(self):
        raise NotImplemented("Implement selection method")


class ClusterBased(SelectionMethod):
    def __init__(self, info: ClustersInfo) -> None:
        self.info: ClustersInfo = info

    def select(self):
        class_labels = self.info.get_class_labels()
        windows = []
        for label in class_labels:
            for cluster_id in self.info.get_clusters_of_labels(label):
                windows.extend(self.info.best_windows_to_cluster(cluster_id))
        return windows
