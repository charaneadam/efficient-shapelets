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


class DataTransformation(BaseModel):
    fit_time = FloatField()
    transform_time = FloatField()
    n_shapelets = IntegerField()
    dataset = ForeignKeyField(Dataset, backref="transformations")
    method = ForeignKeyField(SelectionMethod, backref="transformations")


class DataTransformationProblem(BaseModel):
    dataset = ForeignKeyField(Dataset, backref="transformations")
    method = ForeignKeyField(SelectionMethod, backref="transformations")


class Classification(BaseModel):
    accuracy = FloatField()
    f1 = FloatField()
    train_time = FloatField()
    test_time = FloatField()
    classifier = ForeignKeyField(Classifier, backref="classifications")
    data = ForeignKeyField(DataTransformation, backref="classifications")


class ClassificationProblem(BaseModel):
    # classifier = ForeignKeyField(Classification, backref="problems")
    dataset = ForeignKeyField(Dataset, backref="classif_problems")
    method = ForeignKeyField(SelectionMethod, backref="classif_problems")
