VOLUME_NAME=shapelets_pgdata
IMAGE_NAME=shapelets_image
CONTAINER_NAME=shapelets


create_volume_if_not_exist(){
  podman volume exists $VOLUME_NAME
  if [[ $? -ne 0 ]]; then
    podman volume create $VOLUME_NAME
  fi
}

build_image(){
  create_volume_if_not_exist
  podman build -t $IMAGE_NAME ./
}

build_image_if_not_exist(){
  podman image exists $IMAGE_NAME
  if [[ $? -ne 0 ]]; then
    build_image
  fi
}

run_container(){
  build_image_if_not_exist
  podman run --replace --rm \
    --name $CONTAINER_NAME \
    -p 5432:5432 -p 8888:8888 -p 8501:8501 \
    -v $VOLUME_NAME:/var/lib/postgresql/data:z \
    -v .:/code:z \
    -d $IMAGE_NAME
}

run_container_if_off(){
  if [[ $(podman ps | grep $CONTAINER_NAME | wc -l) -eq 0 ]]; then
    run_container
  fi
}

if [[ $# -ne 0 ]]; then
  if [[ $1 == "start" ]]; then
    run_container_if_off
  elif [[ $1 == "stop" ]]; then
      podman container stop $CONTAINER_NAME
  elif [[ $1 == "build" ]]; then
    build_image
    run_container
  elif [[ $1 == "bash" ]]; then
    run_container_if_off
    podman exec -it $CONTAINER_NAME bash
  elif [[ $1 == "psql" ]]; then
    run_container_if_off
    podman exec -it $CONTAINER_NAME psql -U postgres -d shapelets
  elif [[ $1 == "run" ]]; then
    run_container_if_off
    podman exec $CONTAINER_NAME bash /scripts/run.sh $2
  elif [[ $1 == "benchmark" ]]; then
    podman exec $CONTAINER_NAME python -m $(echo $(find src/benchmarks/*.py | fzf) | cut -d '.' -f1 | tr '/' '.')
    
  else
    echo "Wrong argument, run using: ./container [start|stop|build|bash|psql|run]]"
  fi
fi
