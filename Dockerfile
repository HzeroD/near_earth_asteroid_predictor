FROM python:3.13-slim AS os

WORKDIR /app

ARG INFERENCE_LOG_BUCKET="$MONITORING_BUCKET"
ENV INFERENCE_LOG_BUCKET="$MONITORING_BUCKET"

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts/batch_load_inference_logs.py ./scripts/batch_load_inference_logs.py
COPY scripts/publish_drift_reports.py ./scripts/publish_drift_reports.py

RUN pip install --upgrade pip

RUN pip install --no-cache-dir .
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn near_earth_asteroid_predictor.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
