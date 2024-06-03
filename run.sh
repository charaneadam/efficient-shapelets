#!/bin/sh

METADATA_EXISTS=$(psql -h localhost -U postgres -d shapelets -t -c \
  "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='metadata';")
if [[ "$METADATA_EXISTS" -eq 0 ]]; then
  METADATA=https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/DataSummary.csv
  curl $METADATA -o /tmp/metadata.csv
  cat /tmp/metadata.csv | \
    awk -F"," '{if ($7 != "Vary") print $1,$2,$3,$4,$5,$6,$7}' \
    > /tmp/clean.csv
  psql -h localhost -U postgres -f /scripts/UCR.sql
fi

if [[ $1 == "jupyter" ]]; then
  if [[ "$(jupyter server list | grep 8888 | wc -l)" -eq 0 ]]; then
    nohup jupyter lab --allow-root \
      --NotebookApp.token='' --NotebookApp.password='' \
      --ip 0.0.0.0 &> jupyter_nohup.out &
  else
    echo "Jupyter notebook already running."
  fi
elif [[ $1 == "webapp" ]] then
  if [[ "$(ps -Al | grep streamlit | wc -l)" -eq 0 ]]; then
    nohup webapp run /code/webapp.py &> webapp_nohup.out &
  else
    echo "Webapp already running."
  fi
else
  echo "Wrong argument, syntax : ./container run [jupyter|webapp]"
fi
