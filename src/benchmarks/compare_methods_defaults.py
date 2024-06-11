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

from src.storage import Dataset as DBDataset
from src.storage import Classifier as DBClassifier
from src.storage import SelectionMethod as DBSelectionMethod
from src.storage import DataTransformation as DBDataTransformation
from src.storage import Classification as DBClassification
from src.storage import ClassificationProblem as DBClassifProblem
from src.storage import DataTransformationProblem as DBTransformProblem
from src.storage import LabelPrecRecall as DBLabelPrecRecall


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
    labels = list(set(y_tr))
    precision, recall, _, _ = precision_recall(y_pred, y_te, labels=labels)

    info["models"][model_name] = {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "accuracy": acc,
        "f1": f1,
        "labels": labels,
        "precision": precision,
        "recall": recall,
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


def _benchmark_method_dataset(dataset_name, method_name, method_params={}):
    print(f"\t{dataset_name} ... ", end="")
    info = {}
    data = Data(dataset_name)
    X_tr, y_tr, X_te, y_te = transform_dataset(data, method_name, method_params, info)
    classify_dataset(X_tr, y_tr, X_te, y_te, info)
    print("done.")
    return info


def save_transformation_from_bench(result, db_dataset, db_method):
    db_transformation = DBDataTransformation.create(
        fit_time=result["fit_time"],
        transform_time=result["transform_time"],
        n_shapelets=result["n_shapelets"],
        dataset=db_dataset,
        method=db_method,
    )
    return db_transformation


def save_models_from_bench(result, db_transformation):
    for classifier_name, content in result["models"].items():
        db_classifier = DBClassifier.get(DBClassifier.name == classifier_name)
        db_classification = DBClassification.create(
            accuracy=content["accuracy"],
            f1=content["f1"],
            train_time=content["fit_time"],
            test_time=content["predict_time"],
            classifier=db_classifier,
            data=db_transformation,
        )
        zipped = zip(content["labels"], content["precision"], content["recall"])
        for label, prec, recall in zipped:
            # print(label, prec, recall, db_classification.id)
            DBLabelPrecRecall.create(
                label=label,
                precision=prec,
                recall=recall,
                classification=db_classification,
            )


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
    db_method = DBSelectionMethod.get(DBSelectionMethod.name == method_name)
    for dataset_name in get_datasets_names(method_name):
        db_dataset = DBDataset.get(DBDataset.name == dataset_name)
        try:
            result = _benchmark_method_dataset(dataset_name, method_name, method_params)
        except DataFailure as msg:
            print(f"Data failure: {msg}")
            db_dataset.problematic = True
            db_dataset.save()
        except TransformationFailrue as msg:
            print(f"Transformation failure: {msg}")
            DBTransformProblem.create(dataset=db_dataset, method=db_method)
        except ClassificationFailure as msg:
            print(f"Classification failure: {msg}")
            save_transformation_from_bench(result, db_dataset, db_method)
            DBClassifProblem.create(dataset=db_dataset, method=db_method)

        else:
            db_transformation = save_transformation_from_bench(
                result, db_dataset, db_method
            )
            save_models_from_bench(result, db_transformation)
    print("Done.")


def benchmark_all():
    for method in get_method_names():
        benchmark_method(method)


if __name__ == "__main__":
    benchmark_all()
