FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      mlflow==2.19.0 \
      numpy==1.26.4 \
      pandas==2.2.3 \
      scikit-learn==1.5.2 \
      matplotlib==3.9.2 \
      seaborn==0.13.2 \
      xgboost==2.1.3

COPY MLproject conda.yaml modelling.py modelling_tuning.py ./
COPY dataset_preprocessing ./dataset_preprocessing

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["mlflow", "run", ".", "--env-manager=local"]
