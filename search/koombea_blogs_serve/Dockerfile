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

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG conda_env=python38
ARG py_ver=3.8.10

RUN conda create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} && \
    conda clean --all -f -y

COPY requirements.txt .
COPY requirements-problems.txt .

RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install -U pip
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install -r requirements.txt
RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m spacy download en_core_web_sm
RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m spacy download es_core_news_sm
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install -r requirements-problems.txt
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install multi-model-server sagemaker-inference
RUN conda install -p "${CONDA_DIR}/envs/${conda_env}" -c pytorch faiss-cpu -y && \
    conda clean --all -f -y

# Copy redis config
COPY redis.conf /etc/redis/redis.conf
# Copy nginx config
COPY nginx.conf /opt/ml/input/config/nginx.conf 
# Copy .pem
COPY dataBaseKey.pem /opt/ml/input/config/dataBaseKey.pem

ENV PATH "${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV ${conda_env}
ENV PATH="/opt/ml/code:${PATH}"

# Copy code files
COPY koombea_serve koombea_serve
COPY serve serve
COPY cache.py cache.py

# make files executable
# RUN chmod +x serve cache.py
