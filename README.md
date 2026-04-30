# Near Earth Asteroid Predictor

This project uses a dataset of more than 40,000 near-Earth asteroids (NEAs) to train machine learning models for three prediction tasks:

- `Hazard Classification`: predict whether an asteroid is potentially hazardous from orbital and physical features
- `Miss Distance`: predict the asteroid's Earth miss distance
- `Size Estimation`: estimate diameter from absolute magnitude and albedo-related inputs

The trained models are packaged behind a FastAPI service and prepared for deployment to Google Cloud Platform (GCP).

## Project Goals

The long-term goal is to expose a cloud-hosted prediction API that serves all three models from a single application. The service is intended to support:

- model inference through HTTP endpoints
- health checks for deployment monitoring
- containerized deployment with Docker
- CI-based build validation for GCP delivery

## Planned Workflow

The outline below shows the expected project flow, starting with the hazard classification task as the initial baseline.

### 1. Data Ingestion and Feature Engineering

- load `data/near_earth_asteroids_2025.csv` into a pandas DataFrame
- remove redundant features
- prepare numeric and categorical features for modeling
- apply scaling and one-hot encoding during training

### 2. Model Training

- use stratified k-fold cross-validation for the imbalanced `pha` target
- use `GridSearchCV` to compare candidate models and tune hyperparameters
- persist the best model and preprocessing artifacts for inference

Estimated time: `2 days`  
With added testing and logging: `3 days`

### 3. API Service and Artifacts

- serve predictions with FastAPI
- load saved model artifacts from the application at runtime
- validate requests with Pydantic
- provide these endpoints:
  - `/`
  - `/predict`
  - `/health`

Estimated time: `2 days`

### 4. Docker and GCP Deployment

- package the application as a Docker image
- prepare the service for Google Cloud deployment
- integrate CI for test and build automation

Estimated time: `2 days`

## Estimated Timeline

The estimates above assume a single deployed model for the initial hazard-classification workflow. Since the full project targets three separate prediction tasks, the overall effort may roughly double in the heavier phases.

Estimated completion time: `14 days` worst case

That estimate is intentionally conservative. Some work, especially API, deployment, and CI setup, can be shared across all three models and may not scale linearly.

## Repository Notes

This repository already includes:

- training code in [train.py](/home/merkis/macaw_ml/near_earth_asteroid_predictor/train.py)
- the API entrypoint in [main.py](/home/merkis/macaw_ml/near_earth_asteroid_predictor/main.py)
- serialized model artifacts in [artifacts](/home/merkis/macaw_ml/near_earth_asteroid_predictor/artifacts)
- CI configuration in [.github/workflows/ci.yml](/home/merkis/macaw_ml/near_earth_asteroid_predictor/.github/workflows/ci.yml)
- containerization files in [Dockerfile](/home/merkis/macaw_ml/near_earth_asteroid_predictor/Dockerfile) and [docker-compose.yml](/home/merkis/macaw_ml/near_earth_asteroid_predictor/docker-compose.yml)
