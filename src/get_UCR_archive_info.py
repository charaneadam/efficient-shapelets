import sqlite3
import pandas as pd


PAPER_DATASETS = [
    "ArrowHead",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DistalPhalanxOutlineCorrect",
    "Earthquakes",
    "ECG200",
    "ElectricDevices",
    "Fish",
    "GunPoint",
    "Ham",
    "Herring",
    "InsectWingbeatSound",
    "Lightning2",
    "Lightning7",
    "Meat",
    "MiddlePhalanxOutlineCorrect",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "RefrigerationDevices",
    "ShapeletSim",
    "Strawberry",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoPatterns",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "Worms",
    "WormsTwoClass",
    "Yoga",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPointAgeSpan",
    "GunPointOldVersusYoung",
    "InsectEPGRegularTrain",
    "MixedShapesSmallTrain",
    "PowerCons",
]

INFO_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
DB = sqlite3.connect("resources/results.db")
df = pd.read_csv(INFO_URL).rename(columns=lambda x: x.strip())
__cast_columns = {'ID': 'int', 'Length': 'int',
                  'Train': 'int', 'Test': 'int', 'Class': 'int'}
__same_length_mask = df.Length != "Vary"
__selected_columns = ['ID', 'Name', 'Length', 'Train', 'Test', 'Class']
df = df[__same_length_mask].astype(__cast_columns)[__selected_columns]
df = df[df.Name.isin(PAPER_DATASETS)]
df.to_sql(name="ucr_info", con=DB, index=False, if_exists="replace")
