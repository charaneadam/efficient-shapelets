import sqlite3
import pandas as pd

INFO_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
DB = sqlite3.connect("resources/results.db")
df = pd.read_csv(INFO_URL).rename(columns=lambda x: x.strip())
__cast_columns = {'ID': 'int', 'Length': 'int',
                  'Train': 'int', 'Test': 'int', 'Class': 'int'}
__same_length_mask = df.Length != "Vary"
__selected_columns = ['ID', 'Name', 'Length', 'Train', 'Test', 'Class']
df = df[__same_length_mask].astype(__cast_columns)[__selected_columns]
df.to_sql(name="ucr_info", con=DB, index=False, if_exists="replace")
