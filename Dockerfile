FROM python:3.13-slim AS os

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir .

COPY main.py .

FROM google/cloud-sdk
RUN gcloud config set account merkis.ruiz1992@gmail.com
RUN gcloud auth login --no-browser
RUN mkdir -p ./artifacts && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/best_model.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/best_model_columntransformer.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/moid_best_model.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/column_transformer_moid.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/best_model_abs_mag.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/column_transformer_abs_mag.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



