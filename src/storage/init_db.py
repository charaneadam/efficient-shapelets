from src.storage.database import db
from src.storage.data import init_ucr_metadata
from src.benchmarks.windows_evaluation.db import init_windows_tables


if __name__ == "__main__":
    init_ucr_metadata(db, mark_ts_with_nan=True)
    init_windows_tables(db)
