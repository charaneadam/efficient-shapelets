from time import perf_counter
import pandas as pd

from src.exceptions import DataFailure
from src.storage.database import fix_engine

from src.extraction_methods import FixedLength, VariableLength, Centroids, FSS


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
    8: {"class": Centroids, "params": {}, "name": "Clustering"},
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


def extract(data, dataset_id, method_id):
    start = perf_counter()
    method = ExtractionMethod(data, method_id)
    method.extract()
    end = perf_counter()
    extraction_time = end - start

    with fix_engine.connect() as conn:
        query = f"""INSERT INTO extraction_time VALUES
        ({dataset_id}, {method_id}, {extraction_time})"""
        conn.execute(text(query))
        conn.commit()

    positions = method.candidates_positions()
    dfs = []
    for label in positions.keys():
        df = pd.DataFrame(positions[label], columns=["ts", "first", "last"])
        df["label"] = label
        df["dataset"] = dataset_id
        df["method"] = method_id
        dfs.append(df)
    df = pd.concat(dfs)
    candidates = df[["dataset", "method", "ts", "label", "first", "last"]]
    candidates.to_sql("candidates", fix_engine, if_exists="append", index=False)
    return candidates


if __name__ == "__main__":
    from sqlalchemy import text
    from src.storage.data import get_datasets_info, Data

    _create_table_candidates = """CREATE TABLE IF NOT EXISTS candidates
    (id SERIAL PRIMARY KEY, dataset INT, method INT, ts INT,
     label INT, first INT, last INT)"""
    _create_table_extraction_time = """CREATE TABLE IF NOT EXISTS extraction_time
     (dataset BIGINT, method INT, time DECIMAL)"""
    with fix_engine.connect() as conn:
        conn.execute(text(_create_table_candidates))
        conn.execute(text(_create_table_extraction_time))
        conn.commit()

    info_df = get_datasets_info()
    info_df = info_df[info_df.length >= 60]
    for dataset_id, dataset_name in zip(info_df.id, info_df.name):
        try:
            data = Data(dataset_name)
        except DataFailure as e:
            print(e)
            continue
        # for method_id in EXTRACTION_METHODS.keys():
        for method_id in [8]:
            extract(data, dataset_id, method_id)
