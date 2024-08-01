import numpy as np

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
                positions[i][0] = mapper[positions[i][0]]
            self.candidates[label] = candidates
            self.candidates_positions[label] = positions


class Centroids:
    def __init__(self, dataset_info) -> None:
        self.data_info = dataset_info

    def generate_candidates(self):
        pass


EXTRACTION_METHODS = {
    0: {
        "class": FixedLength,
        "params": {"window_length": 0.05},
        "name": "Fixed length 5%",
    },
    1: {
        "class": FixedLength,
        "params": {"window_length": 0.1},
        "name": "Fixed length 10%",
    },
    2: {
        "class": FixedLength,
        "params": {"window_length": 0.2},
        "name": "Fixed length 20%",
    },
    3: {
        "class": FixedLength,
        "params": {"window_length": 0.3},
        "name": "Fixed length 30%",
    },
    4: {
        "class": FixedLength,
        "params": {"window_length": 0.4},
        "name": "Fixed length 40%",
    },
    5: {
        "class": FixedLength,
        "params": {"window_length": 0.5},
        "name": "Fixed length 50%",
    },
    6: {"class": VariableLength, "params": {}, "name": "Random variable length"},
    7: {"class": FSS, "params": {}, "name": "Fast Shapelet Transform"},
    # 8: {"class": Centroids, "params": {"number_centroids": 0.1}, "name": "Clustering"},
}


class ExtractionMethod:
    def __init__(self, data, method_id) -> None:
        self.id = method_id
        method = EXTRACTION_METHODS[method_id]
        self.method = method["class"](data, **method["params"])

    def extract(self):
        self.method.generate_candidates()

    def candidates(self):
        return self.method.candidates

    def candidates_positions(self):
        return self.method.candidates_positions

