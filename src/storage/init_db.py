from src.exceptions import DataFailure
from src.storage.database import Dataset, SelectionMethod
from src.storage.database import db
from src.storage.database import (
    ClassificationKmeans,
    ClassificationProblem,
    Classifier,
    DataMethod,
    DataTransformationProblem,
    KmeansParameters,
    Result,
    TimeAccF1,
    TransformationInfo,
    PrecisionRecall
)
from .data import get_dataset


def init_ucr_metadata(mark_ts_with_nan=False):
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
        dataset = Dataset.create(**dict(zip(names, row)))
        if mark_ts_with_nan:
            try:
                get_dataset(dataset.name, train=True)
                get_dataset(dataset.name, train=True)
            except DataFailure:
                dataset.missing_values = True
                dataset.save()


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


if __name__ == "__main__":
    init_ucr_metadata(mark_ts_with_nan=True)
    create_tables()
