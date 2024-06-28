from peewee import CharField, FloatField, ForeignKeyField, IntegerField
from src.storage.database import BaseModel
from src.benchmarks.windows_evaluation.db import WindowsEvaluation


class ClassificationModel(BaseModel):
    name = CharField(unique=True)


class ScoringMethod(BaseModel):
    name = CharField(unique=True)


class ClassificationResult(BaseModel):
    skip_size = IntegerField()
    top_K = IntegerField()
    model = ForeignKeyField(ClassificationModel)
    windows_evaluation = ForeignKeyField(WindowsEvaluation)
    scoring_method = ForeignKeyField(ScoringMethod)


class TimeAccF1(BaseModel):
    accuracy = FloatField()
    f1 = FloatField()
    train_time = FloatField()
    test_time = FloatField()
    result = ForeignKeyField(ClassificationResult, backref="acc_f1")


class PrecisionRecall(BaseModel):
    label = IntegerField()
    precision = FloatField()
    recall = FloatField()
    result = ForeignKeyField(ClassificationResult, backref="prec_recall")
