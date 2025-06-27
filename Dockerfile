FROM jupyter/datascience-notebook:python-3.11

# Install additional ML and tracking packages
RUN pip install --no-cache-dir \
    prophet==1.1.6 \
    catboost==1.2 \
    xgboost==3.0.2 \
    lightgbm==4.6.0 \
    tensorflow==2.19.0 \
    mlflow==2.22.0 \
    streamlit==1.35.0 \
    clickhouse-connect==0.7.7
