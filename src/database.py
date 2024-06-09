from peewee import PostgresqlDatabase
from peewee import Model, ForeignKeyField
from peewee import CharField, IntegerField, FloatField, DoubleField

db = PostgresqlDatabase(
    database="shapelets", user="postgres", host="localhost", port=5432
)


class BaseModel(Model):
    class Meta:
        database = db


class Dataset(BaseModel):
    data_type = CharField()
    name = CharField()
    train = IntegerField()
    test = IntegerField()
    n_classes = IntegerField()
    length = IntegerField()


class Method(BaseModel):
    method_name = CharField()
    dataset = ForeignKeyField(Dataset, backref="methods")


class MethodParameter(BaseModel):
    name = CharField()
    value = FloatField()
    runtime = DoubleField()
    method = ForeignKeyField(Method, backref="parameters")


class MethodShapelets(BaseModel):
    length = IntegerField()
    number = IntegerField()
    dataset = ForeignKeyField(Dataset, backref="shapelets_properties")
    method = ForeignKeyField(Method, backref="shapelets_properties")


class ModelClassification(BaseModel):
    model_name = CharField()
    accuracy = FloatField()
    f1 = FloatField()
    method_shapelets = ForeignKeyField(MethodShapelets, backref="classifiers")


TABLES = [
    Dataset,
    Method,
    MethodParameter,
    MethodShapelets,
    ModelClassification,
]


def init_ucr_metadata():
    import pandas as pd

    with db:
        Dataset.drop_table(cascade=True)
        Dataset.create_table()

    df = pd.read_csv(
        "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
    )
    df = df[df.Length != "Vary"]
    cols = ["Type", "Name", "Train ", "Test ", "Class", "Length"]
    names = ["data_type", "name", "train", "test", "n_classes", "length"]
    for row in df[cols].values:
        row[-1] = int(row[-1])
        Dataset.create(**dict(zip(names, row)))


if __name__ == "__main__":
    init_ucr_metadata()
