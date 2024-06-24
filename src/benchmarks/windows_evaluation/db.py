import numpy as np
from src.storage.database import BaseModel
from src.storage.data import Dataset
from peewee import IntegerField, CharField, FloatField, ForeignKeyField


# Experiment: Windows evaluation (bruteforce vs clustering)
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


def init_windows_tables(db):
    TABLES = [WindowsEvaluationAproach, WindowEvaluation, WindowsEvaluation]
    db.drop_tables(TABLES)
    db.create_tables(TABLES)
    WindowsEvaluationAproach.create(name="Bruteforce")
    WindowsEvaluationAproach.create(name="Clustering")


# Experiment: Clustering parameters
class ClusteringParametersEvaluation(BaseModel):
    dataset = ForeignKeyField(Dataset, backref="clustering_windows_evaluation")
    window_size = IntegerField()
    skip_size = IntegerField()
    n_centroids = IntegerField()
    n_iterations = IntegerField()
    cluster_time = FloatField()
    evaluation_time = FloatField()
    info_time = FloatField()


class WindowEvaluationClustering(BaseModel):
    silhouette = FloatField()
    silhouette_time = FloatField()
    fstat = FloatField()
    fstat_time = FloatField()
    infogain = FloatField()
    infogain_time = FloatField()
    label = IntegerField()
    popularity = FloatField()
    population_size = FloatField()
    distinct_ts = IntegerField()
    avg_dist_same = FloatField(null=True)
    avg_dist_diff = FloatField(null=True)
    params = ForeignKeyField(ClusteringParametersEvaluation, backref="windows")


def init_clustering_tables(db):
    TABLES = [ClusteringParametersEvaluation, WindowEvaluationClustering]
    db.drop_tables(TABLES)
    db.create_tables(TABLES)


def save_clustering_parameters(
    dataset,
    window_size,
    skip_size,
    n_iterations,
    clustering_time,
    centroids_evaluation,
    evaluation_time,
    centroids_info,
    info_time,
):
    n_centroids = centroids_info.shape[0]

    params_id = ClusteringParametersEvaluation.insert(
        dataset=dataset,
        window_size=window_size,
        skip_size=skip_size,
        n_centroids=n_centroids,
        n_iterations=n_iterations,
        cluster_time=clustering_time,
        evaluation_time=evaluation_time,
        info_time=info_time,
    ).execute()
    silhouettes = centroids_evaluation[:, 0]
    silhouettes_time = centroids_evaluation[:, 1]
    fstats = centroids_evaluation[:, 2]
    fstats_time = centroids_evaluation[:, 3]
    infogains = centroids_evaluation[:, 4]
    infogains_time = centroids_evaluation[:, 5]

    population_size = centroids_evaluation[:, 0]
    popularity = centroids_evaluation[:, 1]
    distint_ts = centroids_evaluation[:, 2]
    avg_dist_same = centroids_evaluation[:, 3]
    avg_dist_diff = centroids_evaluation[:, 4]
    centroids_labels = centroids_evaluation[:, 5]

    WindowEvaluationClustering.insert_many(
        list(
            zip(
                silhouettes,
                silhouettes_time,
                fstats,
                fstats_time,
                infogains,
                infogains_time,
                centroids_labels,
                population_size,
                popularity,
                distint_ts,
                avg_dist_same,
                avg_dist_diff,
                [params_id] * len(silhouettes),
            )
        ),
        fields=[
            WindowEvaluationClustering.silhouette,
            WindowEvaluationClustering.silhouette_time,
            WindowEvaluationClustering.fstat,
            WindowEvaluationClustering.fstat_time,
            WindowEvaluationClustering.infogain,
            WindowEvaluationClustering.infogain_time,
            WindowEvaluationClustering.label,
            WindowEvaluationClustering.population_size,
            WindowEvaluationClustering.popularity,
            WindowEvaluationClustering.distinct_ts,
            WindowEvaluationClustering.avg_dist_same,
            WindowEvaluationClustering.avg_dist_diff,
            WindowEvaluationClustering.params,
        ],
    ).execute()
