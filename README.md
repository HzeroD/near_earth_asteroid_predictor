# Near Earth Asteroid Predictor

This project trains and serves machine learning models for near-Earth asteroid prediction tasks. It uses a dataset of more than 40,000 asteroid records and exposes the trained models through a FastAPI application prepared for containerized deployment on Google Cloud Platform.

## What the service predicts

- `Hazard Classification`: whether an asteroid is potentially hazardous
- `Miss Distance`: the asteroid's minimum orbit intersection distance with Earth
- `Size Estimation`: diameter-related predictions based on magnitude and related features

## Repository layout

The repository now follows a more standard application structure:

```text
.
├── .github/workflows/cicd.yml
├── data/
├── docs/
│   └── CICD_QUICK_START.md
├── notebooks/
│   └── train.ipynb
├── scripts/
│   └── train.py
├── src/
│   └── near_earth_asteroid_predictor/
│       ├── __init__.py
│       └── api.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── README.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── pytest.ini
```

## Application structure

- application code lives in [src/near_earth_asteroid_predictor/api.py](/home/merkis/macaw_ml/near_earth_asteroid_predictor/src/near_earth_asteroid_predictor/api.py)
- model training code lives in [scripts/train.py](/home/merkis/macaw_ml/near_earth_asteroid_predictor/scripts/train.py)
- CI/CD setup notes live in [docs/CICD_QUICK_START.md](/home/merkis/macaw_ml/near_earth_asteroid_predictor/docs/CICD_QUICK_START.md)
- API tests live in [tests/test_api.py](/home/merkis/macaw_ml/near_earth_asteroid_predictor/tests/test_api.py)

## Local development

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

Run the API locally:

```bash
uvicorn near_earth_asteroid_predictor.api:app --reload
```

Run tests:

```bash
pytest
```

Build the container locally:

```bash
docker build -t near-earth-asteroid-predictor .
```

## CI/CD workflow

The GitHub Actions workflow is defined in [.github/workflows/cicd.yml](/home/merkis/macaw_ml/near_earth_asteroid_predictor/.github/workflows/cicd.yml).

It follows this deployment path:

1. Pull requests into `main` run tests and container build validation.
2. Pushes to `develop` run tests, build the image, and push a SHA-tagged image to Artifact Registry.
3. Pushes to `main` run tests, build the image, push the SHA-tagged image, tag `latest`, and deploy that revision to Cloud Run.

This lets the repository model a common team workflow:

- feature branches for isolated work
- pull requests for review and CI checks
- `main` as the production deployment branch

## Deployment notes

- model artifacts are downloaded from Google Cloud Storage during the workflow before the Docker image is built
- container images are pushed to Google Artifact Registry
- production deployment targets Google Cloud Run
- authentication is handled through Workload Identity Federation rather than long-lived service account keys

## Project status

The project now has:

- a packaged FastAPI application
- endpoint tests for the prediction API
- Docker-based runtime packaging
- CI for validation
- CD for Cloud Run deployment from `main`
