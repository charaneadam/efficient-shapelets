import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = Path(BASE_PATH).parent / "data"
DATASETS_PATHS = [x[0] for x in os.walk(DATA_PATH)]
DATASETS_NAMES = list(map(lambda x: x.split("/")[-1], DATASETS_PATHS))
DATASETS = dict(zip(DATASETS_NAMES, DATASETS_PATHS))


from src.methods import (
    RandomDilatedShapeletTransform,
    RandomShapeletTransform,
    NearestNeighborTransform,
    LearningShapeletsTransform,
)

SELECTION_METHODS = {
    "Random dilated shapelets": RandomDilatedShapeletTransform,
    "Random shapelets": RandomShapeletTransform,
    "Learning shapelets": LearningShapeletsTransform,
    "Nearest neighbor": NearestNeighborTransform,
}

CLASSIFIERS = {
    "Logistic Regression": (LogisticRegression, {"solver": "liblinear"}),
    "Linear SVM": (SVC, {"kernel": "linear"}),
    "RBF SVM": (SVC, {"gamma": 2, "C": 1}),
    "Gaussian Process": (GaussianProcessClassifier, {"kernel": 1.0 * RBF(1.0)}),
    "QDA": (QuadraticDiscriminantAnalysis, {}),
    "Decision Tree": (DecisionTreeClassifier, {"max_depth": 5}),
    "Random Forest": (RandomForestClassifier, {"max_depth": 5, "n_estimators": 10}),
    "Ada Boost": (AdaBoostClassifier, {"algorithm": "SAMME"}),
}
