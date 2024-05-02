import pandas as pd
import faiss

# TODO: Local imports have to be fixed later, as for now, experiments and code
# changes are being done mainly on jupyter.
from data import Data


class NearestNeighborShapelets:
    def __init__(
        self, data: Data, n_neighbors: int = 100, threshold: float = 0.7
    ) -> None:
        self.windows = data.get_sliding_windows().astype("float32")
        self.data: Data = data
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.df: pd.DataFrame | None = None
        self.init_index()

    def init_index(self):
        self.index = faiss.IndexFlatL2(self.data.window_size)
        self.index.add(self.windows)
        self.distances, self.indices = self.index.search(self.windows, self.n_neighbors)

    def get_shapelets_ids(self):
        windows_labels = self.data.get_windows_labels()
        keep = []
        for i in range(self.data.n_windows):
            window_info = self.data.windows_labels_and_covered_ts(self.indices[i])
            self._ts_covered = None
            neighbors_labels = window_info[0]
            n_ts_covered = len(window_info[1][windows_labels[i]])
            ts_id = i // (self.data._train.ts_length - self.data.window_size + 1)
            population_fraction = (
                sum(neighbors_labels == windows_labels[i]) / self.n_neighbors
            )
            if population_fraction >= self.threshold:
                keep.append(
                    [i, population_fraction, windows_labels[i], n_ts_covered, ts_id]
                )
        self.df = pd.DataFrame(
            sorted(keep, key=lambda x: x[1], reverse=True),
            columns=["window", "popularity", "label", "ts_covered", "ts_id"],
        )

        return self.df.window.values
