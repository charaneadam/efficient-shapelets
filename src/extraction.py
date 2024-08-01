from time import perf_counter
import pandas as pd

from src.exceptions import DataFailure
from src.storage.database import fix_engine

from src.extraction_methods import ExtractionMethod, EXTRACTION_METHODS


def extract(data, dataset_id, method_id):
    start = perf_counter()
    method = ExtractionMethod(data, method_id)
    method.extract()
    end = perf_counter()
    positions = method.candidates_positions()
    dfs = []
    for label in positions.keys():
        df = pd.DataFrame(positions[label], columns=["ts", "start", "end"])
        df["label"] = label
        df["dataset"] = dataset_id
        df["method"] = method_id
        dfs.append(df)
    df = pd.concat(dfs)
    candidates = df[["dataset", "method", "ts", "label", "start", "end"]]
    candidates.to_sql("candidates", fix_engine, if_exists="append", index=False)


if __name__ == "__main__":
    from src.storage.data import get_datasets_info, Data

    info_df = get_datasets_info()
    info_df = info_df[info_df.length >= 60]
    for dataset_id, dataset_name in zip(info_df.id, info_df.name):
        try:
            data = Data(dataset_name)
        except DataFailure as e:
            print(e)
            continue
        for method_id in EXTRACTION_METHODS.keys():
            extract(data, dataset_id, method_id)
