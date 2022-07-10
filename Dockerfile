FROM jupyter/scipy-notebook

USER root
RUN pip install joblib
RUN apt-get update && apt-get install -y jq


RUN mkdir model raw_data processed_data results


ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RAW_DATA_FILE=iris.csv
ENV RESULTS_DIR=/home/jovyan/results


COPY iris.csv ./raw_data/iris.csv
COPY train.py ./train.py
COPY test.py ./test.py