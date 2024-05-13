from time import perf_counter
from sklearn.metrics import accuracy_score
from src.methods import SelectionMethod
from src.classifiers import CLASSIFIERS, Classifier
from .data import Data


def _transform(data, method_name, method_params={}):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test
    method = SelectionMethod(method_name)
    method.set_parameters(method_params)
    method.fit(X_train, y_train)

    X_tr = method.transform(X_train)
    X_te = method.transform(X_test)
    return X_tr, y_train, X_te, y_test


def _classify(X_tr, y_tr, X_te, y_te, model_name, info):
    model = Classifier(model_name)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_pred, y_te)
    info["models"][model_name] = acc


def transform_dataset(data: Data, method_name, params, info):
    start = perf_counter()
    X_tr, y_train, X_te, y_test = _transform(data, method_name, params)
    end = perf_counter()
    elapsed_time = end - start
    num_shapelets = X_tr.shape[1]
    info["time"] = elapsed_time
    info["n_shapelets"] = num_shapelets
    return X_tr, y_train, X_te, y_test


def classify_dataset(X_tr, y_train, X_te, y_test, info):
    info["models"] = dict()
    for model_name in CLASSIFIERS.keys():
        _classify(X_tr, y_train, X_te, y_test, model_name, info)


def run_dataset(dataset_name):
    dataset_info = dict()
    try:
        data = Data(dataset_name)
    except:
        return
    for method_name in SELECTION_METHODS.keys():
        try:
            method_info = dict()
            X_tr, y_train, X_te, y_test = transform_dataset(
                data, method_name, method_info
            )
            classify_dataset(X_tr, y_train, X_te, y_test, method_info)
            dataset_info[method_name] = method_info
        except:
            pass
    return dataset_info


def run():
    results = dict()
    datasets_names = ["CBF", "GunPoint", "ArrowHead", "Beef", "BME"]
    for dataset_name in datasets_names:
        dataset_info = run_dataset(dataset_name)
        results[dataset_name] = dataset_info
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=4))
