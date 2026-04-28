FROM python:3.13-slim AS os

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir .

COPY main.py .

FROM google/cloud-sdk
RUN gcloud config set account merkis.ruiz1992@gmail.com
RUN gcloud auth login --remote-bootstrap="https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=32555940559.apps.googleusercontent.com&scope=openid+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fappengine.admin+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fsqlservice.login+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcompute+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth&state=JUqCU7x8mkPuwhsQsT6pTzCZ6J2Sft&access_type=offline&code_challenge=VSnRSo4pv1HsT8qd4IBo7bT-ElufFOEmG-EC4ITaSto&code_challenge_method=S256&token_usage=remote"
RUN mkdir -p ./artifacts && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/best_model.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/best_model_columntransformer.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/moid_best_model.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/column_transformer_moid.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/best_model_abs_mag.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com && \
    gcloud storage cp gs://project-3e6b348d-e2ae-4a47-9af_cloudbuild/artifacts/column_transformer_abs_mag.pkl ./artifacts/ --impersonate-service-account=github-actions@project-3e6b348d-e2ae-4a47-9af.iam.gserviceaccount.com

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



