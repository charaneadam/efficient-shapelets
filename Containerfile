FROM docker.io/postgres:16

ENV POSTGRES_PASSWORD pass
ENV POSTGRES_DB shapelets

RUN apt-get -y update
RUN apt-get -y install curl unzip procps bzip2 wget vim fzf

# Download and extract UCR archive in /data
RUN mkdir /data
RUN curl https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip\
  -o /data/data.zip && unzip -P someone /data/data.zip -d /data/
RUN mv /data/UCRArchive_2018/* /data/ && rmdir /data/UCRArchive_2018/

# Install micromamba in /micromamba
RUN mkdir /micromamba
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/1.4.2 | \
  tar --directory /micromamba -xvj 

# Install python environment with micromamba
COPY conda_requirements.yaml /tmp/env.yaml
RUN /micromamba/bin/micromamba env create -f /tmp/env.yaml -y

# Add some links to binaries folder, to avoid typing whole path of programs
RUN ln -s /root/micromamba/envs/shapelets/bin/streamlit /usr/bin/webapp
RUN ln -s /root/micromamba/envs/shapelets/bin/jupyter /usr/bin/jupyter
RUN ln -s /root/micromamba/envs/shapelets/bin/python /usr/bin/python
RUN ln -s /micromamba/bin/micromamba /usr/bin/micromamba

# Activate python environment by default when using bash
RUN echo 'eval "$(micromamba shell hook --shell=bash)"' > /root/.bashrc
RUN echo "micromamba activate shapelets" >> /root/.bashrc

# add command runner to /scripts
COPY run.sh /scripts/run.sh

# Make /code default directory (to be mounted from the host)
VOLUME /code
WORKDIR /code

# Expose ports for postgre, streamlit and jupyter
EXPOSE 5432
EXPOSE 8501
EXPOSE 8888
