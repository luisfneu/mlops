FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.162

EXPOSE 5000
