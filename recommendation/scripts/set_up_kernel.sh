#! /usr/bin/env bash
conda_env=python38
py_ver=3.8.10
CONDA_DIR=$HOME/anaconda3


conda create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} && \
    conda clean --all -f -y

"${CONDA_DIR}/envs/${conda_env}/bin/pip" install -U pip
"${CONDA_DIR}/envs/${conda_env}/bin/pip" install ipykernel
"${CONDA_DIR}/envs/${conda_env}/bin/pip" install -r ./requirements/requirements-notebook.txt

"${CONDA_DIR}/envs/${conda_env}/bin/python" -m spacy download en_core_web_sm
"${CONDA_DIR}/envs/${conda_env}/bin/python" -m spacy download es_core_news_sm
"${CONDA_DIR}/envs/${conda_env}/bin/python" -m nltk.downloader stopwords
"${CONDA_DIR}/envs/${conda_env}/bin/pip" install -r ./requirements/requirements-problems.txt

conda install -p "${CONDA_DIR}/envs/${conda_env}" -c pytorch faiss-cpu -y && \
    conda clean --all -f -y
    
"${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name $conda_env --display-name "$conda_env my_env"
