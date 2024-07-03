import pandas as pd
from sqlalchemy import inspect
from src.benchmarks.classification.utils import transform, _classify
from src.benchmarks.get_experiment import get_datasets
from src.classifiers import CLASSIFIERS_NAMES
from src.storage.data import Data
from src.storage.database import engine


def candidates_df(dataset_name):
    query = f"""SELECT silhouette, fstat, gain, label, ts_id, start, length
                FROM fixed_length_candidates
                WHERE dataset='{dataset_name}'
            """
    fixed_lengths = pd.read_sql(query, engine)
    fixed_lengths.drop_duplicates(
        subset=["ts_id", "start", "length"], keep="first", inplace=True
    )
    return fixed_lengths


def top_K_shapelets_info(df, method, K):
    candidates = []
    selected_columns = ["ts_id", "start", "length"]
    df.sort_values(by=method, ascending=False, inplace=True)
    for label in df.label.unique():
        view = df[(df.label == label)]
        candidates.extend(view.head(K)[selected_columns].values)
    return candidates


def topK_shapelets(df, method, K, data):
    info = top_K_shapelets_info(df, method, K)
    shapelets = []
    for ts_id, start, length in info:
        cand = data.X_train[ts_id][start: start + length]
        cand = (cand - cand.mean()) / cand.std()
        shapelets.append(cand)
    return shapelets


def classify(df, data, method, K):
    shapelets = topK_shapelets(df, method, K, data)
    X_tr, X_te = transform(data, shapelets)
    accuracies = {}
    for clf_name in CLASSIFIERS_NAMES:
        res = _classify(clf_name, X_tr, data.y_train, X_te, data.y_test)
        fit_time, predict_time, acc, f1, labels, precision, recall = res
        accuracies[clf_name] = acc
    return accuracies


def compare(dataset_name):
    df = candidates_df(dataset_name)
    data = Data(dataset_name)
    results = []
    for method in ["silhouette", "gain", "fstat"]:
        for K in [3, 5, 10, 20, 50, 100]:
            accuracies = classify(df, data, method, K)
            models_accuracies = [
                accuracies.get(clf_name, None) for clf_name in CLASSIFIERS_NAMES
            ]
            result = [dataset_name, method, K] + models_accuracies
            results.append(result)
    return results


def run():
    # datasets = get_datasets()
    columns = ["dataset", "method", "K_shapelets"] + CLASSIFIERS_NAMES
    # inspector = inspect(engine)
    TABLE_NAME = "fixed_variable_lengths"
    # if inspector.has_table(TABLE_NAME):
        # current_df = pd.read_sql(TABLE_NAME, engine)
        # computed = set(current_df.dataset.unique())
    # else:
        # computed = set()

    datasets = ["CBF","GunPoint"]
    for dataset in datasets:
        # if dataset.length < 60 or dataset.name in computed:
            # continue
        try:
            results = compare(dataset)
            df = pd.DataFrame(results, columns=columns)
            df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)
        except:
            print(f"Error happened with dataset {dataset}")


if __name__ == "__main__":
    run()
