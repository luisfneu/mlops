FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
       "scikit-learn>=1.5" \
       "pandas>=2.2" \
       "numpy>=1.26" \
       "pyarrow>=16.0" \
       "joblib>=1.4" \
       "mlflow>=2.15,<3.0" \
       "boto3>=1.34" \
       "fastapi>=0.115" \
       "uvicorn[standard]>=0.30" \
       "pydantic>=2.8" \
       "prometheus-client>=0.20" \
       "prometheus-fastapi-instrumentator>=7.0" \
       "pandera>=0.20" \
       "evidently>=0.4.30,<0.5" \
       "python-dotenv>=1.0" \
       "typer>=0.12" \
       "rich>=13.7" \
       "httpx>=0.27"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
