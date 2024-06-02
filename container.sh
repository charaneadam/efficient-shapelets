VOLUME_NAME=shapelets_pgdata
IMAGE_NAME=shapelets_image
CONTAINER_NAME=shapelets

podman volume exists $VOLUME_NAME
if [[ $? -ne 0 ]]; then
  podman volume create $VOLUME_NAME
fi

podman image exists $IMAGE_NAME
if [[ $? -ne 0 ]]; then
  podman build -t $IMAGE_NAME ./
fi

run_container(){
  podman run --replace --rm \
    --name $CONTAINER_NAME \
    -p 5432:5432 -p 8888:8888 -p 8501:8501 \
    -v $VOLUME_NAME:/var/lib/postgresql/data:z \
    -v ./src:/code:z \
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
    if [[ $(podman ps | grep $CONTAINER_NAME | wc -l) -eq 1 ]]; then
      podman container stop shapelets
    fi
  elif [[ $1 == "build" ]]; then
    podman build -t $IMAGE_NAME ./
    run_container
  elif [[ $1 == "bash" ]]; then
    run_container_if_off
    podman exec -it shapelets bash
  elif [[ $1 == "run" ]]; then
    run_container_if_off
    podman exec shapelets bash /scripts/run.sh $2
  else
    echo "Wrong argument, run using: ./container [build|bash|run]]"
  fi
fi
