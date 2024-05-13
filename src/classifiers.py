from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from src.helpers.exceptions import ClassifierDoesNotExist

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression,
    "Linear SVM": SVC,
    "RBF SVM": SVC,
    "Gaussian Process": GaussianProcessClassifier,
    "QDA": QuadraticDiscriminantAnalysis,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Ada Boost": AdaBoostClassifier,
}

DEFAULT_CLASSIFIERS_PARAMS = {
    "Logistic Regression": {"solver": "liblinear"},
    "Linear SVM": {"kernel": "linear"},
    "RBF SVM": {"gamma": 2, "C": 1},
    "Gaussian Process": {"kernel": 1.0 * RBF(1.0)},
    "QDA": {},
    "Decision Tree": {"max_depth": 5},
    "Random Forest": {"max_depth": 5, "n_estimators": 10},
    "Ada Boost": {"algorithm": "SAMME"},
}


class Classifier:
    def __init__(self, classifier_name):
        if classifier_name not in CLASSIFIERS:
            raise ClassifierDoesNotExist(classifier_name)
        self._alg = CLASSIFIERS[classifier_name]
        params = DEFAULT_CLASSIFIERS_PARAMS.get(classifier_name, {})
        self._alg = self._alg(**params)

    def fit(self, X, y):
        self._alg.fit(X, y)

    def predict(self, X):
        return self._alg.predict(X)
