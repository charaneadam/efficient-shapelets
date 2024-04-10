from data import Data
from selection import LIST_OF_ALGORITHMS, ClustersInfo
from selection.methods import ClusterBased


class Experiment:
    def __init__(self, description, data, algorithm) -> None:
        self.description = description
        self.selection_method = ClustersInfo(data, algorithm)
        self.selection = ClusterBased(self.selection_method)

    def info_df(self):
        return self.selection_method.info()

    def get_shapelets(self):
        self.selection.select()

    def __repr__(self) -> str:
        return self.description


def experiment_parser(metadata) -> Experiment:
    description = metadata["description"]

    data = Data(**metadata["data"])

    algorithm_name = next(iter(metadata["algorithm"].keys()))
    algorithm_params = metadata["algorithm"][algorithm_name]
    algorithm = LIST_OF_ALGORITHMS[algorithm_name](**algorithm_params)

    return Experiment(description, data, algorithm)
