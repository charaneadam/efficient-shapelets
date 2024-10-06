from src.computations import evaluate_candidate
from src.config import DB
from src.data import Data


def init_evaluations_database(cursor):
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS evaluations
            (
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
    query = """SELECT DISTINCT extraction_id FROM candidates
                WHERE candidate_id NOT IN (
                    SELECT DISTINCT candidate_id FROM evaluations
                );"""
    return cursor.execute(query).fetchall()


def data_by_extraction_id(extraction_id, cursor):
    query = f"""SELECT Name from ucr_info
                WHERE ID=(SELECT dataset FROM extractions
                            WHERE extraction_id={extraction_id});"""
    dataset_name = cursor.execute(query).fetchone()[0]
    return Data(dataset_name)


def data_and_candidate_info(extraction_id, cursor):
    query = f"""SELECT candidate_id, ts, start, end
                    FROM candidates
                    WHERE extraction_id={extraction_id}"""
    candidates_info = cursor.execute(query).fetchall()
    data = data_by_extraction_id(extraction_id, cursor)
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


def evaluate_extraction(extraction_id, cursor):
    data, candidates_info = data_and_candidate_info(extraction_id, cursor)
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
        evaluate_extraction(extraction[0], cursor)
