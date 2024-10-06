import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
import faiss

from src.exceptions import NormalizationFailure
from src.data import Data
from src.fss import FastShapeletCandidates


def normalize(candidate):
    try:
        return (candidate - np.mean(candidate)) / np.std(candidate)
    except:
        raise NormalizationFailure


class FSS:
    def __init__(self, data) -> None:
        self.data: Data = data
        self.candidates = dict()
        self.candidates_positions = dict()

    def generate_candidates(self):
        X_train = self.data.X_train
        # n_lfdp and std are the recommended parameters of the authors of FSS
        n_lfdp = int(X_train.shape[1] * 0.05 + 2)
        std = 0.5
        fss = FastShapeletCandidates(n_lfdp, std)
        for label in self.data.labels:
            ts_ids_by_label = np.where(self.data.y_train == label)[0]
            mapper = {idx: id for idx, id in enumerate(ts_ids_by_label)}
            positions, candidates = fss.transform(X_train[ts_ids_by_label])
            for i in range(len(positions)):
                candidates[i] = normalize(candidates[i])
                # remap ts_idx to ts_id (positions[i][0])
                positions[i][0] = mapper[positions[i][0]]
            self.candidates[label] = candidates
            self.candidates_positions[label] = positions


class Centroids:
    def __init__(self, data) -> None:
        self.data: Data = data
        self.candidates = dict()
        self.candidates_positions = dict()

    def generate_candidates(self):
        for label in self.data.labels:
            ids_ts_label = np.where(self.data.y_train == label)[0]
            data_label_view = self.data.X_train[ids_ts_label]

            self.candidates_positions[label] = []
            self.candidates[label] = []
            length_percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            for length in map(
                lambda x: int(x * self.data.ts_length), length_percentages
            ):
                data_label_windows = sliding_window_view(
                    data_label_view, (1, length)
                )
                n_ts, n_windows_per_ts, _, _ = data_label_windows.shape
                # assert n_ts == sum(self.data.y_train == label)
                # assert n_windows_per_ts == self.data.ts_length - length + 1

                n_total_windows = n_ts * n_windows_per_ts
                n_centroids = int(np.sqrt(n_total_windows))
                windows_view = data_label_windows.reshape(
                    n_total_windows, length)
                windows_view = StandardScaler().fit_transform(windows_view.T).T

                km = faiss.Kmeans(length, n_centroids, niter=5)
                km.train(windows_view)
                dists, indices = km.index.search(windows_view, 1)
                indices = indices.reshape(-1)
                dists = dists.reshape(-1)

                for centroid_index in range(n_centroids):
                    centroid_windows = np.where(indices == centroid_index)[0]
                    if len(centroid_windows) == 0:
                        continue
                    index_window_minimal_distance = centroid_windows[
                        np.argmin(dists[centroid_windows])
                    ]
                    ts_idx = index_window_minimal_distance // n_windows_per_ts
                    ts_id = ids_ts_label[ts_idx]
                    # assert label == self.data.y_train[ts_id]
                    start = index_window_minimal_distance % n_windows_per_ts
                    end = start + length
                    self.candidates_positions[label].append(
                        [ts_id, start, end])
                    self.candidates[label].append(
                        windows_view[index_window_minimal_distance]
                    )
