import pandas as pd

INFO_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
INFO_PATH = "resources/datasets_info.parquet"
df = pd.read_csv(INFO_URL).rename(columns=lambda x: x.strip())
__cast_columns = {'ID': 'int16', 'Length': 'int16',
                  'Train': 'int16', 'Test': 'int16', 'Class': 'int16'}
__same_length_mask = df.Length != "Vary"
__selected_columns = ['ID', 'Name', 'Length', 'Train', 'Test', 'Class']
df = df[__same_length_mask].astype(__cast_columns)[__selected_columns]
df.to_parquet(path=INFO_PATH, index=False, compression='gzip')
