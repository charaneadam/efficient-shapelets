from .learning_shapelets import LearningShapeletsTransform
from .nn_shapelets import NearestNeighborTransform
from .randomShapelets import RandomShapeletTransform, RandomDilatedShapeletTransform
from .fss import FastShapeletSelectionTransform
from .clustering import KmeansTransform


from src.exceptions import TransformationDoesNotExist


SELECTION_METHODS = {
    "RandomDilatedShapelets": RandomDilatedShapeletTransform,
    "RandomShapelets": RandomShapeletTransform,
    "LearningShapelets": LearningShapeletsTransform,
    "FastShapeletSelection": FastShapeletSelectionTransform,
    "NearestNeighbor": NearestNeighborTransform,
    "Kmeans": KmeansTransform,
}


class SelectionMethod:
    def __init__(self, method_name) -> None:
        if method_name not in SELECTION_METHODS:
            raise TransformationDoesNotExist(method_name)
        self.name: str = method_name
        self._alg = SELECTION_METHODS[method_name]

    def set_parameters(self, params={}):
        self._alg = self._alg(**params)

    def fit(self, X, y):
        self._alg.fit(X, y)

    def transform(self, X):
        return self._alg.transform(X)
