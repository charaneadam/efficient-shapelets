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


class WindowEvaluation(BaseModel):
    silhouette = FloatField()
    silhouette_time = FloatField()
    fstat = FloatField()
    fstat_time = FloatField()
    infogain = FloatField()
    infogain_time = FloatField()
    window = IntegerField()  # index of the window, or label if it is a centroid
    evaluation = ForeignKeyField(WindowsEvaluation, backref="windows")


def init_windows_tables(db):
    TABLES = [WindowsEvaluationAproach, WindowEvaluation, WindowsEvaluation]
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
    silhouettes = results[:, 0]
    silhouettes_time = results[:, 1]
    fstats = results[:, 2]
    fstats_time = results[:, 3]
    infogains = results[:, 4]
    infogains_time = results[:, 5]
    WindowEvaluation.insert_many(
        list(
            zip(
                silhouettes,
                silhouettes_time,
                fstats,
                fstats_time,
                infogains,
                infogains_time,
                labels,
                [windows_evaluation_id] * len(results),
            )
        ),
        fields=[
            WindowEvaluation.silhouette,
            WindowEvaluation.silhouette_time,
            WindowEvaluation.fstat,
            WindowEvaluation.fstat_time,
            WindowEvaluation.infogain,
            WindowEvaluation.infogain_time,
            WindowEvaluation.window,
            WindowEvaluation.evaluation,
        ],
    ).execute()
