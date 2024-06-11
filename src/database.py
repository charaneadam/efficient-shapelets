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


class Classification(BaseModel):
    accuracy = FloatField()
    f1 = FloatField()
    train_time = FloatField()
    test_time = FloatField()
    classifier = ForeignKeyField(Classifier, backref="classifications")
    data = ForeignKeyField(DataTransformation, backref="classifications")


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


def create_tables():
    TABLES = [SelectionMethod, Classifier, DataTransformation, Classification]
    db.drop_tables(TABLES)
    db.create_tables(TABLES)
    insert_method_names()
    insert_classifiers_names()


if __name__ == "__main__":
    init_ucr_metadata()
    create_tables()
