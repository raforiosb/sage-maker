FROM continuumio/miniconda3

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc  \
    mono-mcs \
    build-essential \
    ca-certificates \
    wget \
    pkg-config

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
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install install sagemaker-training
RUN conda install -p "${CONDA_DIR}/envs/${conda_env}" -c pytorch faiss-cpu -y && \
    conda clean --all -f -y

ENV PATH "${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV ${conda_env}

COPY train.py .
COPY koombea_model koombea_model
COPY vars.env .
COPY vars.prod.env .
COPY vars.staging.env .

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train.py