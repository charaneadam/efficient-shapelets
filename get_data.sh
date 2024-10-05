mkdir -p resources
DATA_URL=https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip
curl $DATA_URL -o resources/data.zip
unzip -P someone resources/data.zip -d resources/
python -m src.get_UCR_archive_info
