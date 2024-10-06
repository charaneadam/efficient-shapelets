import numpy as np
import pandas as pd
from time import perf_counter
from src.data import Data
from src.extraction_methods import Centroids, FSS
from src.config import DB

METHODS_NAMES = {"FSS": FSS, "Clustering": Centroids}
METHODS_IDS = {"FSS": 0, "Clustering": 1}


def init_extraction_database(cursor):
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS extractions
            (
                extraction_id INTEGER PRIMARY KEY,
                dataset INT NOT NULL,
                method INT NOT NULL,
                n_candidates INT NOT NULL,
                time REAL NOT NULL
            )
        """)
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS candidates
            (
                candidate_id INTEGER PRIMARY KEY,
                extraction_id INTEGER NOT NULL,
                ts INT NOT NULL,
                start INT NOT NULL,
                end INT NOT NULL
            )
        """)


def get_datasets_info(cursor):
    query = """SELECT ID, Name FROM ucr_info
                WHERE ID NOT IN (SELECT DISTINCT dataset FROM extractions)
                ORDER BY Length*Train*Test;"""
    return cursor.execute(query).fetchall()


def extract(data, method_name):
    """Given a dataset object and an extraction method name, this function
    returns the list of candidates metadata and the runtime needed to extract
    them. Each value in the list has the form: [ts_id, start_pos, end_pos]."""
    start = perf_counter()
    method = METHODS_NAMES[method_name](data)
    method.generate_candidates()
    end = perf_counter()
    positions = np.vstack([v for v in method.candidates_positions.values()])
    return positions, end-start


def save_extraction(dataset_id, method_id, positions, time, cursor):
    n_candidates = positions.shape[0]
    cursor.execute(
        """INSERT INTO extractions(dataset,method,n_candidates,time)
            VALUES(?,?,?,?)""",
        (dataset_id, method_id, n_candidates, time)
    )

    extraction_id = cursor.lastrowid
    info = np.hstack((
        extraction_id * np.ones(n_candidates, dtype="int").reshape(-1, 1),
        positions
    ))
    pd.DataFrame(
        info,
        columns=["extraction_id", "ts", "start", "end"]
    ).to_sql("candidates", DB, index=False, if_exists="append")


def run(cursor):
    datasets_metadata = get_datasets_info()
    for dataset_id, dataset_name in datasets_metadata:
        data = Data(dataset_name)
        for method_name, method_id in METHODS_IDS.items():
            positions, time = extract(data, method_name)
            save_extraction(dataset_id, method_id, positions, time, cursor)


if __name__ == "__main__":
    cursor = DB.cursor()
    init_extraction_database(cursor)
    for dataset_id, dataset_name in get_datasets_info(cursor):
        data = Data(dataset_name)
        for method_name, method_id in METHODS_IDS.items():
            positions, time = extract(data, method_name)
            save_extraction(dataset_id, method_id, positions, time, cursor)
    DB.commit()
