# CI/CD Setup Guide for GCP Deployment

This guide provides step-by-step instructions to set up CI/CD for the Near Earth Asteroid Predictor using GitHub Actions and Google Cloud Platform.

## Prerequisites

- GitHub repository
- GCP project with billing enabled
- `gcloud` CLI installed locally
- Docker installed locally (for testing)

---

## Phase 1: Google Cloud Platform Setup

### 1.1 Create GCP Project

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Create a new project
gcloud projects create $PROJECT_ID --name="Near Earth Asteroid Predictor"

# Set it as active
gcloud config set project $PROJECT_ID
```

### 1.2 Enable Required APIs

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  iam.googleapis.com
```

### 1.3 Create Artifact Registry Repository

```bash
gcloud artifacts repositories create near-earth-asteroid-predictor \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for Near Earth Asteroid Predictor"
```

### 1.4 Create Service Account for CI/CD

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Service Account"

# Get the service account email
export SA_EMAIL=$(gcloud iam service-accounts list \
  --filter="displayName:GitHub Actions Service Account" \
  --format='value(email)')

echo $SA_EMAIL
```

### 1.5 Grant Service Account Permissions

```bash
# Grant Artifact Registry permissions (push/pull images)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA_EMAIL \
  --role=roles/artifactregistry.admin

# Grant Cloud Run permissions (deploy services)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA_EMAIL \
  --role=roles/run.admin

# Grant Service Account User permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA_EMAIL \
  --role=roles/iam.serviceAccountUser
```

### 1.6 Set Up Workload Identity Federation (WIF)

This allows GitHub Actions to authenticate to GCP without storing long-lived credentials.

```bash
# Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --project=$PROJECT_ID \
  --location=global \
  --display-name="GitHub Pool"

# Create Workload Identity Provider
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=$PROJECT_ID \
  --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Get the provider resource name
export PROVIDER_RESOURCE_NAME=$(gcloud iam workload-identity-pools providers describe "github-provider" \
  --project=$PROJECT_ID \
  --location=global \
  --workload-identity-pool="github-pool" \
  --format='value(name)')

echo $PROVIDER_RESOURCE_NAME
```

### 1.7 Connect GitHub Repository to Service Account

```bash
# Replace OWNER and REPO with your GitHub username/org and repository name
export GITHUB_OWNER="your-github-username"
export GITHUB_REPO="near_earth_asteroid_predictor"

# Create service account IAM binding
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --project=$PROJECT_ID \
  --role=roles/iam.workloadIdentityUser \
  --principal="principalSet://iam.googleapis.com/$PROVIDER_RESOURCE_NAME/google/subject"
```

---

## Phase 2: GitHub Repository Secrets Configuration

### 2.1 Add GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

1. **GCP_PROJECT_ID**
   - Value: Your GCP project ID

2. **WIF_PROVIDER**
   - Value: The `PROVIDER_RESOURCE_NAME` from step 1.6
   - Format: `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider`

3. **WIF_SERVICE_ACCOUNT**
   - Value: The service account email from step 1.4
   - Format: `github-actions@PROJECT_ID.iam.gserviceaccount.com`

### 2.2 Verify Secrets

```bash
# List all secrets (without values)
echo "Verify these secrets are set in GitHub:"
echo "- GCP_PROJECT_ID"
echo "- WIF_PROVIDER"
echo "- WIF_SERVICE_ACCOUNT"
```

---

## Phase 3: Cloud Run Configuration

### 3.1 Create Initial Cloud Run Service (Optional - Pipeline will create this)

```bash
# This is optional if you want to set it up manually first
gcloud run deploy near-earth-asteroid-predictor \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 3600 \
  --set-env-vars "LOG_LEVEL=INFO"
```

### 3.2 View Deployed Service

```bash
gcloud run services describe near-earth-asteroid-predictor \
  --platform managed \
  --region $REGION \
  --format='value(status.url)'
```

---

## Phase 4: GitHub Actions Workflow Triggers

### 4.1 Understanding Workflow Behavior

The `.github/workflows/deploy.yml` workflow has three jobs:

1. **test**: Runs on every push and pull request
   - Runs pytest with coverage
   - Uploads coverage to Codecov

2. **build-and-push**: Runs after tests pass
   - Builds Docker image
   - Pushes to Artifact Registry
   - Runs on PRs and main branch

3. **deploy**: Runs after image push (main branch only)
   - Deploys to Cloud Run
   - Only triggers on main branch pushes

### 4.2 Manual Trigger (Optional)

To enable manual workflow dispatch, add to `.github/workflows/deploy.yml`:

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:  # Allows manual trigger
```

---

## Phase 5: Local Testing

### 5.1 Build Docker Image Locally

```bash
docker build -t near-earth-asteroid-predictor:latest .
```

### 5.2 Run Container Locally

```bash
docker run -p 8000:8000 near-earth-asteroid-predictor:latest
```

### 5.3 Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get API docs
curl http://localhost:8000/docs
```

---

## Phase 6: Monitoring and Debugging

### 6.1 View Cloud Build Logs

```bash
gcloud builds log -stream $(gcloud builds list --limit=1 --format='value(id)')
```

### 6.2 View Cloud Run Logs

```bash
gcloud run services describe near-earth-asteroid-predictor \
  --platform managed \
  --region $REGION

# Stream logs
gcloud run services logs read near-earth-asteroid-predictor \
  --platform managed \
  --region $REGION \
  --limit 50 \
  --follow
```

### 6.3 Check GitHub Actions Logs

- Go to your GitHub repository → Actions
- Click on the workflow run to see step-by-step logs

---

## Phase 7: Environment Variables and Configuration

### 7.1 Add Custom Environment Variables to Cloud Run

Edit `.github/workflows/deploy.yml` and modify the Cloud Run deploy step:

```yaml
- name: Deploy to Cloud Run
  run: |
    gcloud run deploy ${{ env.IMAGE }} \
      --image "..." \
      --set-env-vars "LOG_LEVEL=INFO,MODEL_CACHE_SIZE=1000"
```

### 7.2 Add Secrets to Cloud Run (if needed)

Create secrets in Secret Manager:

```bash
echo -n "your-secret-value" | gcloud secrets create my-secret --data-file=-

# Grant Cloud Run service account access
gcloud secrets add-iam-policy-binding my-secret \
  --member=serviceAccount:near-earth-asteroid-predictor@${PROJECT_ID}.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

Update Cloud Run deployment to use secrets:

```yaml
--set-env-vars "SECRET_NAME=projects/PROJECT_ID/secrets/my-secret/latest:MY_SECRET_VAR"
```

---

## Troubleshooting

### Issue: GitHub Actions authentication fails

**Solution**: Verify WIF_PROVIDER and WIF_SERVICE_ACCOUNT secrets are correct:

```bash
gcloud iam workload-identity-pools providers describe "github-provider" \
  --project=$PROJECT_ID \
  --location=global \
  --workload-identity-pool="github-pool" \
  --format='value(name)'
```

### Issue: Docker image push fails

**Solution**: Ensure service account has Artifact Registry permissions:

```bash
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:$SA_EMAIL"
```

### Issue: Cloud Run deployment fails

**Solution**: Check Cloud Run service permissions and quotas:

```bash
gcloud run services list --platform=managed --region=$REGION
```

---

## Next Steps

1. **Add Deployment Stages**: Set up separate dev/staging/prod environments
2. **Add Health Checks**: Configure Cloud Run health checks
3. **Enable CORS**: Update FastAPI to handle CORS if needed
4. **Add API Gateway**: Route traffic through Google Cloud Armor
5. **Set Up Monitoring**: Create Cloud Monitoring alerts
6. **Enable Auto-scaling**: Configure Cloud Run auto-scaling policies

---

## Useful Commands Reference

```bash
# Redeploy latest image
gcloud run deploy near-earth-asteroid-predictor \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/near-earth-asteroid-predictor/near-earth-asteroid-predictor:latest \
  --platform managed --region us-central1

# View service metrics
gcloud run services describe near-earth-asteroid-predictor --platform managed --region us-central1

# Delete service
gcloud run services delete near-earth-asteroid-predictor --platform managed --region us-central1

# View service environment variables
gcloud run services describe near-earth-asteroid-predictor --platform managed --region us-central1 --format='value(spec.template.spec.containers[0].env)'
```
