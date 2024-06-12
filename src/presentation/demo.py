import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.presentation.utils import get_windows, silhouette
from src.storage.data import Data


class Demo:
    def __init__(self, dataset_name, samples_per_class=3):
        self.name = dataset_name
        self._data = Data(dataset_name)
        self._n_ts, self.length = self._data.X_train.shape
        self._labels = list(set(self._data.y_train))
        self._n_labels = len(self._labels)
        self._n_samples = samples_per_class
        self._colors = list(mcolors.TABLEAU_COLORS.keys())
        self._ts_sample: list
        self._window_size: int
        self._evaluated: bool
        self._silhouettes: list

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

    def _plot_label(self, label_idx, axs, plot_shapelets=False):
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
                rng_start = self._silhouettes[label_idx][ax_idx][0][0]
                rng_end = rng_start + self._window_size
                rng = np.arange(rng_start, rng_end)
                ax.plot(rng, ts[rng] + label_idx * 4, color="black")
            ax.get_yaxis().set_visible(False)

    def plot_data(self, plot_shapelets=False):
        fig, axs = plt.subplots(1, self._n_samples, figsize=(15, 3))
        for label_idx in range(self._n_labels):
            self._plot_label(label_idx, axs, plot_shapelets)

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

    def _label_id(self, ts_id):
        return ts_id // self._n_samples

    def _ts_idx(self, ts_id):
        return self._ts_sample[ts_id // self._n_samples][ts_id % self._n_samples]

    def get_same_and_different_ts(self, ts_id):
        same = []
        ts_label_id = self._label_id(ts_id)
        for other_id, other_idx in enumerate(self._ts_sample[ts_label_id]):
            if ts_id == other_id:
                continue
            same.append(other_idx)
        same = self._data.X_train[same]

        other = []
        for label_id in range(self._n_labels):
            if label_id == ts_label_id:
                continue
            other.extend(self._ts_sample[label_id])
        other = self._data.X_train[other]

        return same, other

    def _evaluate_ts_candidates(self, ts_id):
        same, different = self.get_same_and_different_ts(ts_id)
        ts = self._data.X_train[self._ts_idx(ts_id)]
        windows = get_windows(ts, self._window_size)
        silhouettes = [
            (idx, silhouette(window, same, different))
            for idx, window in enumerate(windows)
        ]
        silhouettes.sort(key=lambda x: x[1], reverse=True)
        self._silhouettes[ts_id // self._n_samples].append(silhouettes)

    def evaluate_windows(self, window_size):
        self._silhouettes = [[] for _ in range(self._n_labels)]
        self._window_size = window_size
        for ts_id in range(self._n_samples * self._n_labels):
            self._evaluate_ts_candidates(ts_id)
        self._evaluated = True

    def evaluations_df(self, ts_id):
        ts_label_id = ts_id // self._n_samples
        df = pd.DataFrame(
            self._silhouettes[ts_label_id][ts_id % self._n_samples],
            columns=["start_position", "silhouette"],
        )
        df.set_index("start_position", inplace=True)
        return df
