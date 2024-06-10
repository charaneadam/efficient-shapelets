from time import perf_counter
from sklearn.metrics import accuracy_score, f1_score
from src.methods import SelectionMethod
from src.classifiers import CLASSIFIERS, Classifier
from .data import Data


def _transform(method, X_train, X_test):
    X_tr = method.transform(X_train)
    X_te = method.transform(X_test)
    return X_tr, X_te


def _fit(method_name, method_params, X_train, y_train):
    method = SelectionMethod(method_name)
    method.set_parameters(method_params)
    method.fit(X_train, y_train)
    return method


def _classify(X_tr, y_tr, X_te, y_te, model_name, info):
    model = Classifier(model_name)

    start = perf_counter()
    model.fit(X_tr, y_tr)
    end = perf_counter()
    fit_time = end - start

    start = perf_counter()
    y_pred = model.predict(X_te)
    end = perf_counter()
    predict_time = end - start

    acc = accuracy_score(y_pred, y_te)
    if len(set(y_te)) > 2:
        f1 = f1_score(y_pred, y_te, average="weighted")
    else:
        f1 = f1_score(y_pred, y_te)

    info["models"][model_name] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "accuracy": acc,
        "f1": f1,
    }


def transform_dataset(data: Data, method_name, params, info):
    start = perf_counter()
    method = _fit(method_name, params, data.X_train, data.y_train)
    end = perf_counter()
    fit_time = end - start

    start = perf_counter()
    X_tr, X_te = _transform(method, data.X_train, data.X_test)
    end = perf_counter()
    transform_time = end - start

    num_shapelets = X_tr.shape[1]
    info["fit_time"] = fit_time
    info["transform_time"] = transform_time
    info["n_shapelets"] = num_shapelets
    return X_tr, data.y_train, X_te, data.y_test


def classify_dataset(X_tr, y_train, X_te, y_test, info):
    info["models"] = dict()
    for model_name in CLASSIFIERS.keys():
        _classify(X_tr, y_train, X_te, y_test, model_name, info)
