import numpy as np
import pandas as pd
from time import perf_counter
import duckdb
from src.storage.data import Data
from src.extraction_methods import Centroids, FSS
from src.config import DB

METHODS_NAMES = {"FSS": FSS, "Clustering": Centroids}
METHODS_IDS = {"FSS": 0, "Clustering": 1}


def get_datasets_info():
    query = """
    SELECT Id, Name FROM 'resources/datasets_info.parquet'
        WHERE Name in (SELECT * FROM read_csv('paper_datasets.csv', header=False))
        ORDER BY Train*Length*Test;
    """
    return duckdb.sql(query).fetchall()


def extract(data, method_name):
    """Given a dataset name and an extraction method name, this functions
    returns a list of candidates metadata. Each value in the list has the
    form: [ts_id, start_pos, end_pos]."""
    start = perf_counter()
    method = METHODS_NAMES[method_name](data)
    method.generate_candidates()
    end = perf_counter()
    positions = np.vstack([v for v in method.candidates_positions.values()])
    return positions, end-start


def run(cursor):
    datasets_metadata = get_datasets_info()
    for dataset_id, dataset_name in datasets_metadata:
        data = Data(dataset_name)
        for method_name, method_id in METHODS_IDS.items():
            positions, time = extract(data, method_name)
            n_candidates = positions.shape[0]

            cursor.execute(
                """INSERT INTO extraction(dataset,method,n_candidates,time)
                    VALUES(?,?,?,?)""",
                (dataset_id, method_id, n_candidates, time)
            )
            extrct_id = cursor.lastrowid
            info = np.hstack((
                extrct_id * np.ones(n_candidates, dtype="int").reshape(-1, 1),
                positions
            ))
            pd.DataFrame(
                info,
                columns=["extraction_id", "ts", "start", "end"]
            ).to_sql("candidates", DB, index=False, if_exists="append")


if __name__ == "__main__":
    import sys

    cursor = DB.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS extraction
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
    # run(cursor)
    dataset_name = sys.argv[1]
    method_name = sys.argv[2]
    data = Data(dataset_name)
    positions, time = extract(data, method_name)
    print(f"Dataset {dataset_name} {data.X_train.shape}: {time}(s)")

    # DB.commit()
