from time import perf_counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support as precision_recall,
)
from src.classifiers import CLASSIFIERS, Classifier
from .db import ClassificationModel, ClassificationResult, ScoringMethod, TimeAccF1, PrecisionRecall
from src.storage.database import db


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


def _get_classif_metrics(y_pred, y_te, labels):
    acc = accuracy_score(y_pred, y_te)
    if len(set(y_te)) > 2:
        f1 = f1_score(y_pred, y_te, average="weighted")
    else:
        f1 = f1_score(y_pred, y_te)
    precision, recall, _, _ = precision_recall(y_pred, y_te, labels=labels)
    return acc, f1, labels, precision, recall


def _classify(model_name, X_tr, y_tr, X_te, y_te, labels):
    fit_time, predict_time, y_pred = _get_time_and_predictions(
        model_name, X_tr, y_tr, X_te
    )
    acc, f1, labels, precision, recall = _get_classif_metrics(y_pred, y_te, labels)
    return fit_time, predict_time, acc, f1, labels, precision, recall


def _save(
    model_name,
    scoring_method_name,
    windows_evaluation,
    skip_size,
    k,
    fit_time,
    predict_time,
    acc,
    f1,
    labels,
    precision,
    recall,
):

    model = ClassificationModel.get(ClassificationModel.name == model_name)
    method = ScoringMethod.get(ScoringMethod.name == scoring_method_name)
    result = ClassificationResult.create(
        skip_size=skip_size,
        top_K=k,
        model=model,
        windows_evaluation=windows_evaluation,
        scoring_method = method
    )
    TimeAccF1.create(
        accuracy=acc,
        f1=f1,
        train_time=fit_time,
        test_time=predict_time,
        result=result,
    )
    for l, p, r in zip(labels, precision, recall):
        PrecisionRecall.create(label=l, precision=p, recall=r, result=result)


def insert_classifiers_names():
    for classifier_name in CLASSIFIERS.keys():
        ClassificationModel.create(name=classifier_name)

def insert_scoring_methods():
    methods = ["silhouette", "fstat", "infogain"]
    for method in methods:
        ScoringMethod.insert(name=method).execute()

def init_classification_tables():
    TABLES = [ClassificationModel, ClassificationResult, TimeAccF1, PrecisionRecall, ScoringMethod]
    db.drop_tables(TABLES)
    db.create_tables(TABLES)
    insert_classifiers_names()
    insert_scoring_methods()
