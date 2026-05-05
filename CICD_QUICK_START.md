# CI/CD Implementation Summary

## What Has Been Set Up

This project now has a complete CI/CD pipeline ready for GCP deployment. Here's what was created:

### Files Created:

1. **`.github/workflows/deploy.yml`**
   - GitHub Actions workflow file
   - Automated testing, building, and deployment pipeline
   - Triggers on push to main/develop and pull requests
   - Three-stage pipeline: test → build-and-push → deploy

2. **`cloudbuild.yaml`**
   - Alternative GCP Cloud Build configuration
   - Can be used instead of GitHub Actions if you prefer GCP-native CI/CD
   - Useful for connecting the repository directly to GCP

3. **`.dockerignore`**
   - Optimizes Docker build context
   - Excludes unnecessary files from image

4. **`CI_CD_SETUP.md`** (This file)
   - Complete setup instructions for GCP
   - Step-by-step configuration guide
   - Troubleshooting tips

---

## Quick Start (TL;DR)

### Step 1: GCP Setup (One-time)

```bash
# Set variables
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"

# Create GCP project and enable APIs
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  iam.googleapis.com

# Create Artifact Registry
gcloud artifacts repositories create near-earth-asteroid-predictor \
  --repository-format=docker \
  --location=us-central1

# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Service Account"

export SA_EMAIL=$(gcloud iam service-accounts list \
  --filter="displayName:GitHub Actions Service Account" \
  --format='value(email)')

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA_EMAIL \
  --role=roles/artifactregistry.admin

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA_EMAIL \
  --role=roles/run.admin

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$SA_EMAIL \
  --role=roles/iam.serviceAccountUser

# Set up Workload Identity Federation
gcloud iam workload-identity-pools create "github-pool2" \
  --project=$PROJECT_ID \
  --location=global \
  --display-name="GitHub Pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project=$PROJECT_ID \
  --location=global \
  --workload-identity-pool="github-pool2" \
  --display-name="GitHub Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-condition="attribute.repository=='HzeroD/near_earth_asteroid_prediction'"\
  --attribute-mapping="google.subject=assertion.sub, attribute.aud=assertion.aud" 
   
  

export PROVIDER_RESOURCE_NAME=$(gcloud iam workload-identity-pools providers describe "github-provider" \
  --project=$PROJECT_ID \
  --location=global \
  --workload-identity-pool="github-pool2" \
  --format='value(name)')

gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL   --project=$PROJECT_ID   --role=roles/iam.workloadIdentityUser   --member="principalSet://iam.googleapis.com/projects/64675826330/locations/global/workloadIdentityPools/github
-pool2/attribute.repository/google/subject"
```

### Step 2: GitHub Secrets (One-time)

Go to GitHub Repo → Settings → Secrets → Actions

Add these secrets:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `WIF_PROVIDER` | The output from `$PROVIDER_RESOURCE_NAME` above |
| `WIF_SERVICE_ACCOUNT` | The service account email (`$SA_EMAIL`) |

### Step 3: Push to Main

```bash
git add .
git commit -m "Add CI/CD pipeline"
git push origin main
#

```

That's it! The pipeline will now:
- ✅ Run tests automatically
- ✅ Build Docker image
- ✅ Push to Google Artifact Registry
- ✅ Deploy to Cloud Run

---

## Pipeline Flow

```
Push to GitHub
    ↓
[TEST STAGE]
├─ Install dependencies
├─ Run pytest with coverage
└─ Upload coverage reports
    ↓
[BUILD & PUSH STAGE]
├─ Authenticate to GCP
├─ Build Docker image
└─ Push to Artifact Registry
    ↓
[DEPLOY STAGE] (main branch only)
├─ Deploy to Cloud Run
├─ Configure service
└─ Output service URL
```

---

## How to Use

### Automatic Deployments

1. **Pull Requests**: Tests run automatically. Docker image builds but doesn't deploy.
2. **Merge to Main**: All three stages run (test → build → deploy to Cloud Run)
3. **Other Branches**: Only tests run.

### Manual Deployments

To manually trigger the workflow:
1. Go to GitHub → Actions
2. Select the "CI/CD Pipeline" workflow
3. Click "Run workflow"
4. Select branch and click "Run"

### Monitor Deployments

**GitHub Actions:**
- Navigate to Actions tab in your repo
- Click on a workflow run to see logs

**Cloud Run:**
```bash
# View deployed service
gcloud run services describe near-earth-asteroid-predictor \
  --platform managed --region us-central1

# Stream logs
gcloud run services logs read near-earth-asteroid-predictor \
  --platform managed --region us-central1 --limit 50 --follow

# View metrics
gcloud monitoring dashboards list
```

---

## Key Features

✅ **Automated Testing**: Pytest with coverage reports  
✅ **Secure Authentication**: Workload Identity Federation (no stored credentials)  
✅ **Container Registry**: Google Artifact Registry integration  
✅ **Serverless Deployment**: Cloud Run auto-scaling  
✅ **Cost Effective**: Pay only for what you use  
✅ **Production Ready**: Health checks and proper resource allocation  

---

## Environment Variables

The Cloud Run service is configured with:

- `LOG_LEVEL=INFO`: Logging level for the FastAPI app
- `MEMORY=2Gi`: Memory allocation (adjust as needed)
- `TIMEOUT=3600`: Request timeout in seconds (1 hour)

To add more environment variables, edit `.github/workflows/deploy.yml`:

```yaml
--set-env-vars "LOG_LEVEL=INFO,NEW_VAR=value"
```

---

## Security Best Practices

1. **Secrets Management**: Use Google Secret Manager for sensitive data
2. **IAM**: Service account has minimal required permissions
3. **Image Scanning**: Artifact Registry scans images for vulnerabilities
4. **Network**: Consider enabling VPC Service Controls
5. **Logging**: All deployments are logged in Cloud Logging

---

## Troubleshooting

### Tests fail locally but pass in CI?
- Ensure all dependencies are in `pyproject.toml`
- Check Python version compatibility (3.13)

### Docker build fails?
- Verify all model artifacts exist in `./artifacts/`
- Check `.dockerignore` isn't excluding important files

### Cloud Run deployment times out?
- Check Cloud Run logs: `gcloud run services logs read`
- Increase timeout or memory allocation
- Verify all dependencies are installed correctly

### Can't authenticate to GCP?
- Verify WIF_PROVIDER and WIF_SERVICE_ACCOUNT secrets
- Check GitHub Actions has "id-token: write" permissions
- Ensure service account has correct IAM roles

---

## Next Steps

1. **Test Locally First**:
   ```bash
   python -m pytest
   docker build -t test .
   docker run -p 8000:8000 test
   ```

2. **Enable Branch Protection**: Require tests to pass before merge
   - Go to Settings → Branches → Add rule
   - Require status checks to pass

3. **Add More Stages**: Staging environment before production
   - Duplicate deploy step with different environment

4. **Monitor & Alert**:
   - Set up Cloud Monitoring
   - Create alerts for errors and high latency

5. **API Documentation**:
   - Access at `https://your-cloud-run-url/docs`
   - Automatically generated from FastAPI

---

## Support & Documentation

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Google Cloud Run Docs](https://cloud.google.com/run/docs)
- [Artifact Registry Docs](https://cloud.google.com/artifact-registry/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

**Questions?** See `CI_CD_SETUP.md` for detailed setup instructions.
