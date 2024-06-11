import os
from pathlib import Path


DATA_PATH = Path("/data/")
BASE_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = BASE_PATH / "results"
