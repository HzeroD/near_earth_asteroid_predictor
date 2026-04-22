FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir .

COPY main.py .
COPY ./artifacts/best_model.pkl ./artifacts/best_model.pkl
COPY ./artifacts/best_model_columntransformer.pkl ./artifacts/best_model_columntransformer.pkl

COPY ./artifacts/moid_best_model.pkl ./artifacts/moid_best_model.pkl
COPY ./artifacts/column_transformer_moid.pkl ./artifacts/column_transformer_moid.pkl

COPY ./artifacts/best_model_abs_mag.pkl ./artifacts/best_model_abs_mag.pkl
COPY ./artifacts/column_transformer_abs_mag.pkl ./artifacts/column_transformer_abs_mag.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



