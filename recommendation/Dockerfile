FROM continuumio/miniconda3

WORKDIR /opt/ml/code

# installing essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc  \
    mono-mcs \
    build-essential \
    nginx \
    ca-certificates \
    wget \
    pkg-config

# Installing redis
RUN cd /tmp && \
    wget http://download.redis.io/redis-stable.tar.gz && \
    tar xvzf redis-stable.tar.gz && \
    cd redis-stable && \
    make && \
    make install

ARG conda_env=python38
ARG py_ver=3.8.10
ARG CONDA_DIR=/opt/conda

RUN conda create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} && \
    conda clean --all -f -y

# Copy requiremets to container
COPY requirements .


RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install -U pip
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install -r requirements-notebook.txt

RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m spacy download en_core_web_sm
RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m spacy download es_core_news_sm
RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m nltk.downloader stopwords
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install -r requirements-problems.txt

RUN conda install -p "${CONDA_DIR}/envs/${conda_env}" -c pytorch faiss-cpu -y && \
    conda clean --all -f -y

# Define importants env variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV STAGE=$STAGE

ENV PATH="/opt/ml/code:${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV "${conda_env}"
ENV CONDA_PREFIX "${CONDA_DIR}/envs/${conda_env}"
ENV CONFIG_DIR="/opt/ml/input/config"

# Verbose
RUN echo $STAGE

# Copy redis config
COPY config/redis.conf /etc/redis/redis.conf
# Copy nginx config
COPY config/nginx.conf /opt/ml/input/config/nginx.conf
COPY config/analytics-key.json /opt/ml/input/config/analytics-key.json
COPY config/dataBaseKey.pem /opt/ml/input/config

# Copy project
COPY src /opt/ml/code