from src.storage.database import BaseModel
from src.storage.data import Dataset
from peewee import CharField, TextField, IntegerField, FloatField, ForeignKeyField


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
    data_method = ForeignKeyField(DataMethod, backref="transformation_info")


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


def insert_method_names():
    # from src.methods import SELECTION_METHODS

    # methods_names = SELECTION_METHODS.keys()
    methods_names = [
        "RandomDilatedShapelets",
        "RandomShapelets",
        "LearningShapelets",
        "FastShapeletSelection",
    ]
    for method_name in methods_names:
        SelectionMethod.create(name=method_name)


def insert_classifiers_names():
    from src.classifiers import CLASSIFIERS

    for classifier_name in CLASSIFIERS.keys():
        Classifier.create(name=classifier_name)


def init_methods_tables(db):
    TABLES = [
        SelectionMethod,
        Classifier,
        KmeansParameters,
        DataMethod,
        Result,
        TransformationInfo,
        TimeAccF1,
        PrecisionRecall,
        ClassificationKmeans,
        DataTransformationProblem,
        ClassificationProblem,
    ]
    db.drop_tables(TABLES)
    db.create_tables(TABLES)
    insert_method_names()
    insert_classifiers_names()
