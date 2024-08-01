from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from src.presentation.utils import get_windows, silhouette
from src.storage.data import Data, Windows


class DemoData:
    def __init__(self, dataset_name, samples_per_class, skip_size) -> None:
        self.name = dataset_name
        self._skip_size = skip_size
        self._data = Data(dataset_name)
        self._n_ts, self.length = self._data.X_train.shape
        self._labels = np.array(list(set(self._data.y_train)))
        self._n_labels = len(self._labels)
        self._n_samples = samples_per_class
        self._ts_sample: list
        self._window_size: int
        self._colors = list(mcolors.TABLEAU_COLORS.keys())
        self.sample()

    def sample(self):
        y = self._data.y_train
        self._evaluated = False
        self._ts_sample = [
            np.random.choice(
                np.where(y == label)[0],
                size=self._n_samples,
                replace=True if self._n_samples > sum(y == label) else False,
            )
            for label in self._labels
        ]

    def _label_id(self, ts_id):
        return ts_id // self._n_samples

    def _ts_idx(self, ts_id):
        return self._ts_sample[ts_id // self._n_samples][ts_id % self._n_samples]

    def _plot_label(self, label_idx, axs, plot_shapelets, silhouettes):
        indices = self._ts_sample[label_idx]
        for ax_idx, ts_idx in enumerate(indices):
            ts = self._data.X_train[ts_idx]
            ts = (ts - ts.mean()) / ts.std()
            ax = axs[ax_idx]
            ax.plot(
                ts + label_idx * 4,
                color=self._colors[label_idx],
                label=self._labels[label_idx],
            )
            if plot_shapelets:
                rng_start = silhouettes[label_idx][ax_idx][0][0]
                rng_end = rng_start + self._window_size
                rng = np.arange(rng_start, rng_end)
                ax.plot(rng, ts[rng] + label_idx * 4, color="white")
                ax.plot(
                    rng,
                    ts[rng] + label_idx * 4,
                    color="black",
                    # linestyle="--",
                    linewidth=3,
                )
            ax.get_yaxis().set_visible(False)

    def plot(self, plot_shapelets=False, silhouettes=None):
        fig, axs = plt.subplots(1, self._n_samples, figsize=(15, 3))
        for label_idx in range(self._n_labels):
            self._plot_label(label_idx, axs, plot_shapelets, silhouettes)

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        total = self._n_samples * self._n_labels
        fig.suptitle(f"{self.name}: {total} samples from {self._n_labels} classes")
        fig.tight_layout()
        return fig


class DemoSilhouette:
    def __init__(self, demo_data) -> None:
        self.data: DemoData = demo_data
        self._evaluated: bool
        self._silhouettes: list

    def get_same_and_different_ts(self, ts_id):
        same = []
        ts_label_id = self.data._label_id(ts_id)
        for other_id, other_idx in enumerate(self.data._ts_sample[ts_label_id]):
            if ts_id == other_id:
                continue
            same.append(other_idx)
        same = self.data._data.X_train[same]

        other = []
        for label_id in range(self.data._n_labels):
            if label_id == ts_label_id:
                continue
            other.extend(self.data._ts_sample[label_id])
        other = self.data._data.X_train[other]

        return same, other

    def _evaluate_ts_candidates(self, ts_id, window_size):
        same, different = self.get_same_and_different_ts(ts_id)
        ts = self.data._data.X_train[self.data._ts_idx(ts_id)]
        windows = get_windows(ts, window_size)
        silhouettes = [
            (idx, silhouette(window, same, different))
            for idx, window in enumerate(windows)
        ]
        silhouettes.sort(key=lambda x: x[1], reverse=True)
        self._silhouettes[ts_id // self.data._n_samples].append(silhouettes)

    def evaluate_windows(self, window_size):
        start = perf_counter()
        self._silhouettes = [[] for _ in range(self.data._n_labels)]
        for ts_id in range(self.data._n_samples * self.data._n_labels):
            self._evaluate_ts_candidates(ts_id, window_size)
        self._evaluated = True
        end = perf_counter()
        return end - start

    def evaluations_df(self, ts_id):
        ts_label_id = ts_id // self.data._n_samples
        df = pd.DataFrame(
            self._silhouettes[ts_label_id][ts_id % self.data._n_samples],
            columns=["start_position", "silhouette"],
        )
        df.set_index("start_position", inplace=True)
        return df


class DemoPCAKMeans:
    def __init__(self, silhouette, n_centroids):
        self.n_centroids = n_centroids
        self.silhouette = silhouette
        self.data = self.silhouette.data

    def run_pca_kmeans(self, windows_manager):
        X = self.data._data.X_train[list(chain.from_iterable(self.data._ts_sample))]
        windows = windows_manager.get_windows(X)
        self.pca_windows = PCA(n_components=2).fit_transform(windows)
        ts_ids = (
            np.array(
                list(
                    map(
                        windows_manager.get_ts_index_of_window,
                        np.arange(self.pca_windows.shape[0]),
                    )
                )
            )
            // self.data._n_samples
        )
        labels = np.array(self.data._labels)[np.array(ts_ids)]
        self.km = KMeans(n_clusters=self.n_centroids, random_state=0, n_init="auto")
        self.km.fit(self.pca_windows)
        self.df = pd.DataFrame(
            [self.pca_windows[:, 0], self.pca_windows[:, 1], labels],
            index=["PC1", "PC2", "label"],
        ).T
        self.df["label"] = self.df.label.astype(int)

    def plot(self, with_labels=False):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if with_labels:
            sns.scatterplot(
                self.df,
                x="PC1",
                y="PC2",
                hue="label",
                palette=self.data._colors[: self.data._n_labels],
                ax=ax,
            )
            sns.scatterplot(
                x=self.km.cluster_centers_[:, 0],
                y=self.km.cluster_centers_[:, 1],
                c="black",
                ax=ax,
            )
        else:
            sns.scatterplot(self.df, x="PC1", y="PC2", c="grey", ax=ax)
            sns.scatterplot(
                x=self.km.cluster_centers_[:, 0],
                y=self.km.cluster_centers_[:, 1],
                c="red",
                ax=ax,
            )
        fig.tight_layout()
        return fig


class DemoKmeansSilhouette:
    def __init__(self, demo_data, n_centroids):
        self.n_centroids = n_centroids
        self.data: DemoData = demo_data
        self.km: KMeans
        self.centroids_info: pd.DataFrame

    def get_same_and_different_ts(self, ts_label):
        same = []
        other = []
        for label_id, label in enumerate(self.data._labels):
            indices = self.data._ts_sample[label_id]
            if label == ts_label:
                same.extend(self.data._data.X_train[indices])
            else:
                other.extend(self.data._data.X_train[indices])

        return same, other

    def run_kmeans(self, windows_manager):
        start = perf_counter()
        samples_indices = np.array(list(chain.from_iterable(self.data._ts_sample)))
        X = self.data._data.X_train[samples_indices]
        windows = windows_manager.get_windows(X)
        self.km = KMeans(n_clusters=self.n_centroids, random_state=0, n_init="auto")
        self.km.fit(windows)

        windows_ts_indices = samples_indices[
            list(map(windows_manager.get_ts_index_of_window, range(windows.shape[0])))
        ]
        windows_labels = self.data._data.y_train[windows_ts_indices]

        centroids_info = []

        for centroid_id in range(self.n_centroids):
            centroid_windows_index = self.km.labels_ == centroid_id
            centroid_windows_ids = np.where(centroid_windows_index)
            centroid_windows_ts = windows_ts_indices[centroid_windows_ids]
            centroid_windows_labels = windows_labels[centroid_windows_ids]
            popularity = 0
            population_size = 0
            centroid_label = None
            distinct_ts = 0
            for label in self.data._labels:
                index = centroid_windows_labels == label
                pop = sum(index)
                n_population = centroid_windows_labels.shape[0]
                pop /= n_population
                if pop > popularity:
                    popularity = pop
                    centroid_label = label
                    distinct_ts = len(set(centroid_windows_ts[index]))
                    population_size = n_population
            same, other = self.get_same_and_different_ts(centroid_label)
            centroid_silhouette = silhouette(
                self.km.cluster_centers_[centroid_id], same, other
            )
            info = [
                centroid_silhouette,
                centroid_label,
                popularity,
                population_size,
                distinct_ts,
            ]
            centroids_info.append(info)
        cols = [
            "Silhouette",
            "Assigned label",
            "Popularity",
            "Population size",
            "Distinct TS",
        ]
        self.centroids_info = pd.DataFrame(centroids_info, columns=cols)
        end = perf_counter()
        total_time = end - start
        return total_time


class Demo:
    def __init__(self, dataset_name, samples_per_class=3, skip_size=1):
        self.data = DemoData(dataset_name, samples_per_class, skip_size)
        self.silhouette = DemoSilhouette(self.data)
        self.windows: Windows
        self.pca_kmeans: DemoPCAKMeans
        self.kmeans_silhouette: DemoKmeansSilhouette

    def plot_data(self, plot_shapelets=False):
        return self.data.plot(plot_shapelets, self.silhouette._silhouettes)

    def evaluate_windows(self, window_size):
        self.windows = Windows(window_size, skip_size=1)
        self.data._window_size = window_size
        return self.silhouette.evaluate_windows(window_size)

    def run_pca_kmeans(self, n_centroids):
        self.pca_kmeans = DemoPCAKMeans(self.silhouette, n_centroids)
        self.pca_kmeans.run_pca_kmeans(self.windows)

    def run_kmeans(self):
        self.kmeans_silhouette = DemoKmeansSilhouette(
            self.data, self.pca_kmeans.n_centroids
        )
        return self.kmeans_silhouette.run_kmeans(self.windows)
