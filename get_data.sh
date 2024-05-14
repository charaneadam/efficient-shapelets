#!/bin/sh
 
DATA=https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip
curl $DATA -o data.zip 
unzip -P someone data.zip
mv UCRArchive_2018 data
rm data.zip

METADATA=https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/DataSummary.csv
curl $METADATA -o data/metadata.csv
