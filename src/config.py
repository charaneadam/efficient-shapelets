import os
from pathlib import Path
import sqlite3

DATA_PATH = Path("resources/UCRArchive_2018")
BASE_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = BASE_PATH / "results"


DB = sqlite3.connect("resources/results.db")


NUM_THREADS = 8
NUM_CORES_FOR_CLASSIFICATION = 4
