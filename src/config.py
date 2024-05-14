import os
from pathlib import Path


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = Path(BASE_PATH) / "data"
DATASETS_PATHS = [x[0] for x in os.walk(DATA_PATH)]
DATASETS_NAMES = list(map(lambda x: x.split("/")[-1], DATASETS_PATHS))
DATASETS = dict(zip(DATASETS_NAMES, DATASETS_PATHS))
METADATA_PATH = DATA_PATH / "metadata.csv"
