import os
from pathlib import Path


DATA_PATH = Path("/data/")
BASE_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = BASE_PATH / "results"


NUM_THREADS = 8


LOCAL_ENV = os.environ.get("local", False)
if LOCAL_ENV:
    print("Good")
    SAME_LENGTH_CANDIDATES_TABLE_NAME = os.environ["local"]
    VARIABLE_LENGTH_CANDIDATES_TABLE_NAME = "variable_length_candidates"

    SAME_LENGTH_CLASSIFICATION_TABLE_NAME = "fixed_lengths"
    VARIABLE_CLASSIFICATION_LENGTH_TABLE_NAME = "variable_lengths"


if __name__ == "__main__":
    print(LOCAL_ENV)
