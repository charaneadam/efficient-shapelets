from src.computations import evaluate_candidate
from src.config import DB
from src.data import Data


def init_evaluations_database(cursor):
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS evaluations
            (
                evaluation_id INTEGER PRIMARY KEY,
                candidate_id INTEGER NOT NULL,
                fstat REAL NOT NULL,
                silhouette REAL NOT NULL,
                gain REAL NOT NULL,
                distance_time REAL NOT NULL,
                fstat_time REAL NOT NULL,
                silhouette_time REAL NOT NULL,
                gain_time REAL NOT NULL

            )
        """)


def extractions_metadata(cursor):
    query = """SELECT Name, extraction_id
                FROM extractions
                INNER JOIN ucr_info ON ID=dataset
                ORDER BY n_candidates"""
    return cursor.execute(query).fetchall()


def data_and_candidate_info(extraction_metadata, cursor):
    dataset_name, extraction_id = extraction_metadata
    query = f"""SELECT candidate_id, ts, start, end
                    FROM candidates
                    WHERE extraction_id={extraction_id}"""
    candidates_info = cursor.execute(query).fetchall()
    data = Data(dataset_name)
    return data, candidates_info


def save_evaluations(evaluations, cursor):
    query = """INSERT INTO evaluations
                    (candidate_id,
                        fstat, silhouette, gain,
                        distance_time,
                        fstat_time, silhouette_time, gain_time
                    ) VALUES(?,?,?,?,?,?,?,?)"""
    cursor.executemany(query, evaluations)
    DB.commit()


def evaluate_extraction(extraction_metadata, cursor):
    data, candidates_info = data_and_candidate_info(
        extraction_metadata, cursor)
    evaluations = [
        evaluate_candidate(
            data.X_train,
            data.y_train,
            candidate_info
        )
        for candidate_info in candidates_info
    ]
    save_evaluations(evaluations, cursor)


if __name__ == "__main__":
    cursor = DB.cursor()
    init_evaluations_database(cursor)
    for extraction in extractions_metadata(cursor):
        evaluate_extraction(extraction, cursor)
