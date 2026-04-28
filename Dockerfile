FROM python:3.13-slim AS os

WORKDIR /app

COPY pyproject.toml README.md main.py ./

RUN pip install --upgrade pip

RUN pip install --no-cache-dir .
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

