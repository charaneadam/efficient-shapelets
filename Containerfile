FROM docker.io/postgres:16

RUN apt-get -y update
RUN apt-get -y install curl unzip procps

ENV POSTGRES_PASSWORD pass
ENV POSTGRES_DB shapelets

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /miniconda
RUN /miniconda/bin/conda update -n base conda
RUN /miniconda/bin/conda install -n base conda-libmamba-solver
RUN /miniconda/bin/conda config --set solver libmamba
COPY conda_requirements.yaml /tmp/env.yaml
RUN /miniconda/bin/conda env create -f /tmp/env.yaml

RUN echo "source /miniconda/bin/activate && conda activate shapelets" > /root/.bashrc
RUN ln -s /miniconda/envs/shapelets/bin/streamlit /usr/bin/webapp
RUN ln -s /miniconda/envs/shapelets/bin/jupyter /usr/bin/jupyter

COPY UCR.sql /scripts/UCR.sql
COPY run.sh /scripts/run.sh

VOLUME /code
WORKDIR /code

EXPOSE 5432
EXPOSE 8888
EXPOSE 8501
