from time import perf_counter
from sklearn.metrics import accuracy_score
from src.exceptions import ClassificationFailure, TransformationFailrue
from src.methods import SelectionMethod
from src.classifiers import CLASSIFIERS, Classifier
from src.storage.data import Data


def _transform(data, method_name, method_params={}):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test
    try:
        method = SelectionMethod(method_name)
        method.set_parameters(method_params)
        method.fit(X_train, y_train)
        print("Here")

        X_tr = method.transform(X_train)
        X_te = method.transform(X_test)
        return X_tr, y_train, X_te, y_test
    except:
        raise TransformationFailrue


def _classify(X_tr, y_tr, X_te, y_te, model_name, info):
    try:
        model = Classifier(model_name)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_pred, y_te)
        info["models"][model_name] = acc
    except:
        raise ClassificationFailure


def transform_dataset(data: Data, method_name, params, info):
    start = perf_counter()
    X_tr, y_train, X_te, y_test = _transform(data, method_name, params)
    end = perf_counter()
    elapsed_time = end - start
    num_shapelets = X_tr.shape[1]
    info["time"] = elapsed_time
    info["n_shapelets"] = num_shapelets

    optional_params = params.keys()
    for optional_param in optional_params:
        if optional_param in optional_params:
            info[optional_param] = params[optional_param]
    return X_tr, y_train, X_te, y_test


def classify_dataset(X_tr, y_train, X_te, y_test, info):
    info["models"] = dict()
    for model_name in CLASSIFIERS.keys():
        _classify(X_tr, y_train, X_te, y_test, model_name, info)
