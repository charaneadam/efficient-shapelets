FROM docker.io/mambaorg/micromamba

WORKDIR /shapelets

USER root

RUN apt-get update && apt-get -y install cmake g++ curl unzip

COPY . /shapelets

RUN micromamba env create -f environment.yaml -y

RUN bash get_data.sh

