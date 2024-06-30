from src.storage.database import db_peewee
from src.storage.data import init_ucr_metadata
from src.benchmarks.windows_evaluation.db import init_windows_tables


if __name__ == "__main__":
    init_ucr_metadata(db_peewee, mark_ts_with_nan=True)
    init_windows_tables(db_peewee)
