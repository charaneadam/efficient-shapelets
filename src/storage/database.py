from peewee import PostgresqlDatabase
from peewee import Model, ForeignKeyField
from peewee import CharField, IntegerField, FloatField, TextField, BooleanField

db = PostgresqlDatabase(
    database="shapelets", user="postgres", host="localhost", port=5432
)


class BaseModel(Model):
    class Meta:
        database = db


class Dataset(BaseModel):
    name = CharField(unique=True)
    data_type = CharField()
    train = IntegerField()
    test = IntegerField()
    n_classes = IntegerField()
    length = IntegerField()
    missing_values = BooleanField(default=False)
    problematic = BooleanField(default=False)


class SelectionMethod(BaseModel):
    name = CharField(unique=True)
    description = TextField(null=True)


class Classifier(BaseModel):
    name = CharField(unique=True)
    description = TextField(null=True)


class KmeansParameters(BaseModel):
    window_percentage = IntegerField()
    topk = IntegerField()
    n_centroids = IntegerField()
    n_iter = IntegerField()


class DataMethod(BaseModel):
    dataset = ForeignKeyField(Dataset, backref="data_methods")
    method = ForeignKeyField(SelectionMethod, backref="data_methods")
    kmeans_param = ForeignKeyField(KmeansParameters, null=True, backref="data_methods")


class Result(BaseModel):
    classifier = ForeignKeyField(Classifier, backref="results")
    data_method = ForeignKeyField(DataMethod, backref="results")


class TransformationInfo(BaseModel):
    fit_time = FloatField()
    transform_time = FloatField()
    n_shapelets = IntegerField()
    data_method = ForeignKeyField(DataMethod, backref="info")


class TimeAccF1(BaseModel):
    accuracy = FloatField()
    f1 = FloatField()
    train_time = FloatField()
    test_time = FloatField()
    result = ForeignKeyField(Result, backref="acc_f1")


class PrecisionRecall(BaseModel):
    label = IntegerField()
    precision = FloatField()
    recall = FloatField()
    result = ForeignKeyField(Result, backref="prec_recall")


class ClassificationKmeans(BaseModel):
    result = ForeignKeyField(Result, backref="kmeans_params")


class ClassificationProblem(BaseModel):
    dataset = ForeignKeyField(Dataset, backref="classif_problems")
    method = ForeignKeyField(SelectionMethod, backref="classif_problems")


class DataTransformationProblem(BaseModel):
    dataset = ForeignKeyField(Dataset, backref="transformations")
    method = ForeignKeyField(SelectionMethod, backref="transformations")
