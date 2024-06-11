from time import perf_counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support as precision_recall,
)
from src.methods import SelectionMethod
from src.classifiers import CLASSIFIERS, Classifier

from src.exceptions import DataFailure, TransformationFailrue, ClassificationFailure

from src.storage.data import Data

from src.storage.database import (
    SelectionMethod as DBSelectionMethod,
    Classifier as DBClassifier,
    Dataset as DBDataset,
    DataMethod,
    PrecisionRecall,
    Result,
    TimeAccF1,
    TransformationInfo,
)


def _transform(method, X_train, X_test):
    X_tr = method.transform(X_train)
    X_te = method.transform(X_test)
    return X_tr, X_te


def _fit(method_name, method_params, X_train, y_train):
    method = SelectionMethod(method_name)
    method.set_parameters(method_params)
    method.fit(X_train, y_train)
    return method


def _fit_method_time(method_name, params, X_train, y_train):
    start = perf_counter()
    method = _fit(method_name, params, X_train, y_train)
    end = perf_counter()
    fit_time = end - start
    return method, fit_time


def _transform_time(method, X_train, X_test):
    start = perf_counter()
    X_tr, X_te = _transform(method, X_train, X_test)
    end = perf_counter()
    transform_time = end - start
    return X_tr, X_te, transform_time


def transform_dataset(data: Data, method_name, params, data_method_id):
    X_train, y_train, X_test = data.X_train, data.y_train, data.X_test
    method, fit_time = _fit_method_time(method_name, params, X_train, y_train)
    X_tr, X_te, transform_time = _transform_time(method, X_train, X_test)

    num_shapelets = X_tr.shape[1]

    TransformationInfo.create(
        fit_time=fit_time,
        transform_time=transform_time,
        n_shapelets=num_shapelets,
        data_method_id=data_method_id,
    )

    return X_tr, data.y_train, X_te, data.y_test


def _get_time_and_predictions(model_name, X_tr, y_tr, X_te):
    model = Classifier(model_name)
    start = perf_counter()
    model.fit(X_tr, y_tr)
    end = perf_counter()
    fit_time = end - start

    start = perf_counter()
    y_pred = model.predict(X_te)
    end = perf_counter()
    predict_time = end - start

    return fit_time, predict_time, y_pred


def _get_classif_metrics(y_pred, y_te):
    acc = accuracy_score(y_pred, y_te)

    if len(set(y_te)) > 2:
        f1 = f1_score(y_pred, y_te, average="weighted")
    else:
        f1 = f1_score(y_pred, y_te)

    labels = list(set(y_te))
    precision, recall, _, _ = precision_recall(y_pred, y_te, labels=labels)

    return acc, f1, labels, precision, recall


def _classify_and_store(model_name, X_tr, y_tr, X_te, y_te, model_id, dmid):

    fit_time, predict_time, y_pred = _get_time_and_predictions(
        model_name, X_tr, y_tr, X_te
    )
    acc, f1, labels, precision, recall = _get_classif_metrics(y_pred, y_te)

    result = Result.create(classifier_id=model_id, data_method_id=dmid)

    TimeAccF1.create(
        accuracy=acc,
        f1=f1,
        train_time=fit_time,
        test_time=predict_time,
        result=result,
    )
    for l, p, r in zip(labels, precision, recall):
        PrecisionRecall.create(label=l, precision=p, recall=r, result=result)


def classify_dataset(X_tr, y_train, X_te, y_test, dmid):
    for model_name in CLASSIFIERS.keys():
        mid = DBClassifier.get(DBClassifier.name == model_name).id
        _classify_and_store(model_name, X_tr, y_train, X_te, y_test, mid, dmid)


def _benchmark_method_dataset(dataset, method, method_params={}):
    print(f"\t{dataset.name} ... ", end="")
    data = Data(dataset.name)

    dmid = DataMethod.create(dataset=dataset, method=method).id
    X_tr, y_tr, X_te, y_te = transform_dataset(data, method.name, method_params, dmid)
    classify_dataset(X_tr, y_tr, X_te, y_te, dmid)
    print("done.")


def get_datasets_names(method):
    # TODO
    # given a method name, this function returns list of dataset names
    # that have not been run yet.
    datasets = ["Chinatown", "ItalyPowerDemand", "ECGFiveDays", "GunPoint", "ArrowHead"]
    return datasets


def get_method_names():
    # TODO
    # This function returns list of selection method names that have not been run yet.
    method_names = [
        "RandomShapelets",
        "RandomDilatedShapelets",
        "LearningShapelets",
        "FastShapeletSelection",
    ]
    return method_names


def benchmark_method(method_name, method_params={}):
    print(f"Benchmarking method: {method_name}")
    method = DBSelectionMethod.get(DBSelectionMethod.name == method_name)

    for dataset_name in get_datasets_names(method_name):
        dataset = DBDataset.get(DBDataset.name == dataset_name)

        try:
            result = _benchmark_method_dataset(dataset, method, method_params)
        except DataFailure as msg:
            print(f"Data failure: {msg}")
            dataset.problematic = True
            dataset.save()
        except TransformationFailrue as msg:
            print(f"Transformation failure: {msg}")
        except ClassificationFailure as msg:
            print(f"Classification failure: {msg}")


def benchmark_all():
    for method in get_method_names():
        benchmark_method(method)


if __name__ == "__main__":
    benchmark_all()
