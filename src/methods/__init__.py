from .learning_shapelets import LearningShapeletsTransform
from .nn_shapelets import NearestNeighborTransform
from .randomShapelets import RandomShapeletTransform, RandomDilatedShapeletTransform
from .fss import FastShapeletSelectionTransform
from .clustering import KmeansTransform
from .fss import FastShapeletSelectionTransform


from src.exceptions import TransformationDoesNotExist


SELECTION_METHODS = {
    "Random dilated shapelets": RandomDilatedShapeletTransform,
    "Random shapelets": RandomShapeletTransform,
    "Learning shapelets": LearningShapeletsTransform,
    "Fast Shapelet Selection": FastShapeletSelectionTransform,
    "Nearest neighbor": NearestNeighborTransform,
    "Kmeans": KmeansTransform,
}


class SelectionMethod:
    def __init__(self, method_name) -> None:
        if method_name not in SELECTION_METHODS:
            raise TransformationDoesNotExist(method_name)
        self.name: str = method_name
        self._alg = SELECTION_METHODS[method_name]

    def set_parameters(self, params):
        self._alg = self._alg(**params)

    def fit(self, X, y):
        self._alg.fit(X, y)

    def transform(self, X):
        return self._alg.transform(X)
