from numba import njit
from time import perf_counter
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support as precision_recall,
)
from src.benchmarks.windows_evaluation.utils import distance_numba
from src.classifiers import CLASSIFIERS_MODELS, Classifier
from .db import (
    ClassificationModel,
    ClassificationResult,
    ScoringMethod,
    TimeAccF1,
    PrecisionRecall,
)
from src.storage.database import db_peewee


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
    precision, recall, _, _ = precision_recall(y_pred, y_te)
    return acc, f1, None, precision, recall


def _classify(model_name, X_tr, y_tr, X_te, y_te):
    fit_time, predict_time, y_pred = _get_time_and_predictions(
        model_name, X_tr, y_tr, X_te
    )
    acc, f1, labels, precision, recall = _get_classif_metrics(y_pred, y_te)
    return fit_time, predict_time, acc, f1, labels, precision, recall


@njit(parallel=True)
def _transform(X, shapelets):
    n_ts = X.shape[0]
    n_shapelets = len(shapelets)
    trans = np.zeros((n_ts, n_shapelets))
    for ts_id in range(n_ts):
        ts = X[ts_id]
        for shapelet_id in range(n_shapelets):
            shapelet = shapelets[shapelet_id]
            trans[ts_id, shapelet_id] = distance_numba(ts, shapelet)
    return trans


def transform(data, shapelets):
    X_tr = _transform(data.X_train, shapelets)
    X_te = _transform(data.X_test, shapelets)
    return X_tr, X_te


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
        scoring_method=method,
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
    for classifier_name in CLASSIFIERS_MODELS.keys():
        ClassificationModel.create(name=classifier_name)


def insert_scoring_methods():
    methods = ["silhouette", "fstat", "infogain"]
    for method in methods:
        ScoringMethod.insert(name=method).execute()


def init_classification_tables():
    TABLES = [
        ClassificationModel,
        ClassificationResult,
        TimeAccF1,
        PrecisionRecall,
        ScoringMethod,
    ]
    db_peewee.drop_tables(TABLES)
    db_peewee.create_tables(TABLES)
    insert_classifiers_names()
    insert_scoring_methods()
