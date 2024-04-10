from data import Data
from selection import LIST_OF_ALGORITHMS, ClustersInfo
from selection.methods import ClusterBased
from sklearn.linear_model import LogisticRegression


class Experiment:
    # TODO: Parameters have to come from some experiment metadata class!
    def __init__(self, description, data, algorithm) -> None:
        self.description = description
        self.data: Data = data
        self.selection_method = ClustersInfo(self.data, algorithm)
        # TODO: Initialize the selection mathod and selection somewhere else
        self.selection = ClusterBased(self.selection_method)
        # TODO: Initialize classifier somewhere else and pass it to this class
        self.classifier = LogisticRegression(random_state=0)

    def info_df(self):
        return self.selection_method.info()

    def get_shapelets(self):
        return self.selection.select()

    def classify(self):
        shapelets = self.get_shapelets()
        train = self.data.shapelet_transform(shapelets)
        test = self.data.shapelet_transform(shapelets, train=False)
        self.classifier.fit(train, self.data._train.y)
        preds = self.classifier.predict(test)
        # TODO: Separate the evaluation 
        return sum(preds == self.data._test.y) / len(preds)

    def __repr__(self) -> str:
        return self.description


def experiment_parser(metadata) -> Experiment:
    description = metadata["description"]

    data = Data(**metadata["data"])

    algorithm_name = next(iter(metadata["algorithm"].keys()))
    algorithm_params = metadata["algorithm"][algorithm_name]
    algorithm = LIST_OF_ALGORITHMS[algorithm_name](**algorithm_params)

    return Experiment(description, data, algorithm)
