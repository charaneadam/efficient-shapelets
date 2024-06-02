podman volume exists shapelets_pgdata
if [[ $? -ne 0 ]]; then
  podman volume create shapelets_pgdata
fi

podman image exists shapelets_img
if [[ $? -ne 0 ]]; then
  podman build -t shapelets_img ./
fi

if [[ $(podman ps | grep shapelets | wc -l) -eq 0 ]]; then
  podman run --replace --rm \
    --name shapelets \
    -p 5432:5432 -p 8888:8888 -p 8501:8501 \
    -v shapelets_pgdata:/var/lib/postgresql/data:z \
    -v ./src:/code:z \
    -d shapelets_img
fi

if [[ $# -ne 0 ]]; then
  if [[ $1 == "build" ]]; then
    podman build -t shapelets_img ./
  elif [[ $1 == "bash" ]]; then
    podman exec -it shapelets bash
  elif [[ $1 == "run" ]]; then
    podman exec shapelets bash /scripts/run.sh $2
  else
    echo "Wrong argument, run using: ./container [build|bash|run]]"
  fi
fi
