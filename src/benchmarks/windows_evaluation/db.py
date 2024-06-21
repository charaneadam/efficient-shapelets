import numpy as np
from src.storage.database import BaseModel, Dataset
from peewee import IntegerField, CharField, FloatField, ForeignKeyField


class WindowsEvaluationAproach(BaseModel):
    name = CharField()


class WindowsEvaluation(BaseModel):
    dataset = ForeignKeyField(Dataset, backref="windows_evaluation")
    window_size = IntegerField()
    skip_size = IntegerField()
    approach = ForeignKeyField(WindowsEvaluationAproach, backref="windows_evaluation")
    runtime = FloatField()


class WindowSilhouette(BaseModel):
    silhouette = FloatField()
    window = IntegerField() # index of the window, or label if it is a centroid
    evaluation = ForeignKeyField(WindowsEvaluation, backref="windows")


def init_windows_tables(db):
    TABLES = [WindowsEvaluationAproach, WindowSilhouette, WindowsEvaluation]
    db.drop_tables(TABLES)
    db.create_tables(TABLES)
    WindowsEvaluationAproach.create(name="Bruteforce")
    WindowsEvaluationAproach.create(name="Clustering")


def save(
    dataset_name, method_name, window_size, skip_size, runtime, results, labels=None
):
    dataset_id = Dataset.get(Dataset.name == dataset_name)
    approach_id = WindowsEvaluationAproach.get(
        WindowsEvaluationAproach.name == method_name
    )
    windows_evaluation_id = WindowsEvaluation.insert(
        dataset=dataset_id,
        window_size=window_size,
        skip_size=skip_size,
        approach=approach_id,
        runtime=runtime,
    ).execute()
    if labels is None:
        labels = np.arange(len(results))
    WindowSilhouette.insert_many(
        list(zip(results, labels, [windows_evaluation_id] * len(results))),
        fields=[
            WindowSilhouette.silhouette,
            WindowSilhouette.window,
            WindowSilhouette.evaluation,
        ],
    ).execute()
