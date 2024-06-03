FROM docker.io/postgres:16

RUN apt-get -y update
RUN apt-get -y install curl unzip procps bzip2 wget

ENV POSTGRES_PASSWORD pass
ENV POSTGRES_DB shapelets

COPY conda_requirements.yaml /tmp/env.yaml
RUN mkdir /micromamba
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/1.4.2 | tar --directory /micromamba -xvj 
RUN /micromamba/bin/micromamba env create -f /tmp/env.yaml -y

RUN ln -s /root/micromamba/envs/shapelets/bin/streamlit /usr/bin/webapp
RUN ln -s /root/micromamba/envs/shapelets/bin/jupyter /usr/bin/jupyter
RUN ln -s /micromamba/bin/micromamba /usr/bin/micromamba
RUN echo "micromamba activate shapelets" > /root/.bashrc

COPY UCR.sql /scripts/UCR.sql
COPY run.sh /scripts/run.sh

VOLUME /code
WORKDIR /code

EXPOSE 5432
EXPOSE 8888
EXPOSE 8501
