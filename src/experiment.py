from data import Data
from clustering import LIST_OF_ALGORITHMS, ClustersInfo


class Experiment:
    def __init__(self, metadata):

        self.description = metadata["description"]

        data = Data(**metadata["data"])

        algorithm_name = next(iter(metadata["algorithm"].keys()))
        algorithm_params = metadata["algorithm"][algorithm_name]
        algorithm = LIST_OF_ALGORITHMS[algorithm_name](**algorithm_params)

        self.cluster_info = ClustersInfo(data, algorithm)

    def run(self):
        return self.cluster_info.get_info_df()

    def __repr__(self) -> str:
        return self.description
