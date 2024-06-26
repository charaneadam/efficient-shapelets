from peewee import CharField, FloatField, ForeignKeyField, IntegerField
from src.storage.database import BaseModel
from src.benchmarks.windows_evaluation.db import WindowsEvaluation


class Classifier(BaseModel):
    name = CharField()


class Classification(BaseModel):
    skip_size = IntegerField()
    top_K = IntegerField()
    runtime = FloatField()
    model = ForeignKeyField(Classifier)
    windows_evaluation = ForeignKeyField(WindowsEvaluation)
    accuracy = FloatField()
    f1 = FloatField()
