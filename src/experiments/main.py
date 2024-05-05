from time import perf_counter
from sklearn.metrics import accuracy_score
from .config import CLASSIFIERS
from .config import SELECTION_METHODS
from .data import Data


def _transform(data, Approach, params={}):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test
    app = Approach(**params)
    app.fit(X_train, y_train)

    X_tr = app.transform(X_train)
    X_te = app.transform(X_test)
    return X_tr, y_train, X_te, y_test


def _classify(X_tr, y_tr, X_te, y_te, model_name):
    Model, params = CLASSIFIERS[model_name]
    model = Model(**params)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_pred, y_te)
    print(f"Model name: {model_name}, accuracy: {acc}")


def transform_dataset(data: Data, method_name):
    method = SELECTION_METHODS[method_name]
    start = perf_counter()
    X_tr, y_train, X_te, y_test = _transform(data, method)
    end = perf_counter()
    elapsed_time = end - start
    num_shapelets = X_tr.shape[1]
    print(
        f"{method_name} ran in {elapsed_time}(s) and extracted {num_shapelets} shapelets."
    )
    return X_tr, y_train, X_te, y_test


def classify_dataset(X_tr, y_train, X_te, y_test):
    for model_name in CLASSIFIERS.keys():
        _classify(X_tr, y_train, X_te, y_test, model_name)


def run(dataset_name):
    data = Data(dataset_name)
    for method_name in SELECTION_METHODS.keys():
        X_tr, y_train, X_te, y_test = transform_dataset(data, method_name)
        classify_dataset(X_tr, y_train, X_te, y_test)
        print()


if __name__ == "__main__":
    datasets_names = ["CBF", "GunPoint"]
    for dataset_name in datasets_names:
        print(dataset_name)
        run(dataset_name)
        print()
        print()
        print()
