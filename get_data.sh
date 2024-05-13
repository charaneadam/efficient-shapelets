#!/bin/sh
 
LINK=https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip
curl $LINK -o data.zip 
unzip -P someone data.zip
mv UCRArchive_2018 data
