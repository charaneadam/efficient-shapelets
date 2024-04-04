from data import Data
from selection import LIST_OF_ALGORITHMS, ClustersInfo


class Experiment:
    def __init__(self, description, data, algorithm) -> None:
        self.description = description
        self.selection_method = ClustersInfo(data, algorithm)

    def run(self):
        return self.selection_method.info()

    def __repr__(self) -> str:
        return self.description


def experiment_parser(metadata) -> Experiment:
    description = metadata["description"]

    data = Data(**metadata["data"])

    algorithm_name = next(iter(metadata["algorithm"].keys()))
    algorithm_params = metadata["algorithm"][algorithm_name]
    algorithm = LIST_OF_ALGORITHMS[algorithm_name](**algorithm_params)

    return Experiment(description, data, algorithm)
