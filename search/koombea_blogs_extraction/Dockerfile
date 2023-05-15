# pull image from docker hub
FROM python:3.8.0-slim-buster

# Install system dependencies
RUN apt-get update \
    && apt-get -y install netcat gcc \
    && apt-get clean

# Select workdir
WORKDIR /opt/ml

# copy requirements and install with pip
RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download es_core_news_sm


COPY run.py .
COPY dataBaseKey.pem /opt/ml/input/config/dataBaseKey.pem
COPY koombea_blogs koombea_blogs

# Set environment variables FOR PYTHON EXECUTION
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
