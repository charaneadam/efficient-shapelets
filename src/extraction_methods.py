import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
import faiss

from src.exceptions import NormalizationFailure
from src.storage.data import Data
from src.methods.fss import FastShapeletCandidates


def normalize(candidate):
    try:
        return (candidate - np.mean(candidate)) / np.std(candidate)
    except:
        raise NormalizationFailure


class FixedLength:
    def __init__(self, data: Data, window_length) -> None:
        self.data: Data = data
        self.length: int = int(data.ts_length * window_length)
        self._n_cands_per_class: int = max(300, int(0.2 * self.data.ts_length))

        self.__fail: int = 0
        self.candidates = dict()
        self.candidates_positions = dict()

    def _sample_subsequence_positions(self):
        start_pos = np.random.randint(self.data.ts_length - self.length + 1)
        end_pos = start_pos + self.length
        return start_pos, end_pos

    def _sample_candidate(self, label, ts_ids):
        try:
            start, end = self._sample_subsequence_positions()
            ts_id = np.random.choice(ts_ids)
            candidate = self.data.X_train[ts_id][start:end]
            candidate = normalize(candidate)
            self.candidates[label].append(candidate)
            self.candidates_positions[label].append([ts_id, start, end])
            return 1
        except NormalizationFailure:
            self.__fail += 1
            return 0

    def generate_candidates(self):
        for label in self.data.labels:
            self.candidates[label] = []
            self.candidates_positions[label] = []
            ts_ids = np.where(self.data.y_train == label)[0]
            self.__fail = 0
            n_extracted = 0
            while n_extracted < self._n_cands_per_class and self.__fail < 10:
                n_extracted += self._sample_candidate(label, ts_ids)


class VariableLength:
    def __init__(self, data) -> None:
        self.data: Data = data
        self._n_cands_per_class: int = max(300, int(0.2 * self.data.ts_length))

        self.__fail: int = 0
        self.candidates = dict()
        self.candidates_positions = dict()

    def _sample_subsequence_positions(self):
        ts_length = self.data.ts_length
        start_pos = np.random.randint(int(0.9 * ts_length))
        length = np.random.randint(int(0.05 * ts_length), int(0.7 * ts_length))
        length = max(length, 3)
        end_pos = min(ts_length, start_pos + length)
        return start_pos, end_pos

    def _sample_candidate(self, label, ts_ids):
        try:
            start, end = self._sample_subsequence_positions()
            ts_id = np.random.choice(ts_ids)
            candidate = self.data.X_train[ts_id][start:end]
            candidate = normalize(candidate)
            self.candidates[label].append(candidate)
            self.candidates_positions[label].append([ts_id, start, end])
            return 1
        except NormalizationFailure:
            self.__fail += 1
            return 0

    def generate_candidates(self):
        for label in self.data.labels:
            self.candidates[label] = []
            self.candidates_positions[label] = []
            ts_ids = np.where(self.data.y_train == label)[0]
            self.__fail = 0
            n_extracted = 0
            while n_extracted < self._n_cands_per_class and self.__fail < 10:
                n_extracted += self._sample_candidate(label, ts_ids)


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
            label_positions = np.where(self.data.y_train == label)[0]
            data_label_view = self.data.X_train[label_positions]

            self.candidates_positions[label] = []
            self.candidates[label] = []
            length_percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            for length in map(
                lambda x: int(x * self.data.ts_length), length_percentages
            ):
                data_label_windows = sliding_window_view(
                    data_label_view, (1, length)
                ).squeeze()
                n_ts, n_windows_per_ts, _ = data_label_windows.shape
                assert n_ts == sum(self.data.y_train == label)

                n_total_windows = n_ts * n_windows_per_ts
                n_centroids = int(np.sqrt(n_total_windows))
                windows_view = data_label_windows.reshape(n_total_windows, length)
                windows_view = StandardScaler().fit_transform(windows_view.T).T

                km = faiss.Kmeans(length, n_centroids)
                km.train(windows_view)
                dists, indices = km.index.search(windows_view, 1)
                indices = indices.reshape(-1)
                dists = dists.reshape(-1)

                for centroid_index in range(n_centroids):
                    centroid_windows = np.where(indices == centroid_index)[0]
                    index_window_minimal_distance = centroid_windows[
                        np.argmin(dists[centroid_windows])
                    ]
                    ts_idx = index_window_minimal_distance // n_windows_per_ts
                    ts_id = label_positions[ts_idx]
                    start = index_window_minimal_distance % n_windows_per_ts
                    end = start + length
                    self.candidates_positions[label].append([ts_id, start, end])
                    self.candidates[label].append(
                        windows_view[index_window_minimal_distance]
                    )
