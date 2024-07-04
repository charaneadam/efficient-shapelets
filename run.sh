#!/bin/sh

METADATA_EXISTS=$(psql -h localhost -U postgres -d shapelets -t -c \
  "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='dataset';")
if [[ "$METADATA_EXISTS" -eq 0 ]]; then
  python -m src.storage.init_db
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
    nohup python -m streamlit run src/webapp.py &> webapp_nohup.out &
  else
    echo "Webapp already running."
  fi
else
  echo "Wrong argument, syntax : ./container run [jupyter|webapp]"
fi
