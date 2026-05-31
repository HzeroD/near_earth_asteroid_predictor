# Monitoring Scheduling Runbook

This runbook describes how to automate the monitoring scripts for the Near
Earth Asteroid Predictor project.

## Goal

Run monitoring jobs on a schedule so inference logs and drift reports are
loaded into durable GCP monitoring tables without manual intervention.

The recommended pattern is:

1. Build the project container.
2. Deploy monitoring commands as Cloud Run Jobs.
3. Trigger those jobs with Cloud Scheduler.

## Current Monitoring Commands

Inference log loading:

```bash
uv run python scripts/batch_load_inference_logs.py \
  --project-id "$PROJECT_ID" \
  --bucket "$MONITORING_BUCKET" \
  --dataset near_earth_monitoring \
  --table inference_events \
  --logs logs/*.jsonl
```

Drift report publishing:

```bash
uv run python scripts/publish_drift_reports.py \
  --project-id "$PROJECT_ID" \
  --bucket "$MONITORING_BUCKET" \
  --dataset near_earth_monitoring \
  --report pha=reports/pha_drift.html \
  --report-json pha=reports/pha_drift.json \
  --report moid=reports/moid_drift.html \
  --report-json moid=reports/moid_drift.json \
  --model-version local \
  --window-start 2026-05-28T00:00:00Z \
  --window-end 2026-05-29T00:00:00Z
```

## Important Log Storage Note

The API currently writes inference logs to local JSONL files under `logs/`.
That works locally and in VM-style deployments, but Cloud Run service
filesystems are ephemeral and isolated per container instance.

Before scheduling production log loading, choose one durable log source:

- Write inference logs to GCS.
- Export Cloud Logging entries to BigQuery.
- Run the API on infrastructure with persistent/shared log storage.

Until that choice is made, a scheduled Cloud Run Job can exercise the loader
but may not be able to read the API service's live local logs.

## One-Time GCP Setup

Set environment variables:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPOSITORY="near-earth-asteroid-predictor"
export IMAGE="near-earth-asteroid-predictor"
export MONITORING_BUCKET="your-monitoring-bucket"
export MONITORING_SA="nea-monitoring-jobs@$PROJECT_ID.iam.gserviceaccount.com"
```

Enable required APIs:

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudscheduler.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com
```

Create the monitoring service account:

```bash
gcloud iam service-accounts create nea-monitoring-jobs \
  --display-name="NEA Monitoring Jobs"
```

Grant required IAM roles:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$MONITORING_SA" \
  --role="roles/run.invoker"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$MONITORING_SA" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$MONITORING_SA" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$MONITORING_SA" \
  --role="roles/bigquery.jobUser"
```

Create the monitoring bucket if it does not already exist:

```bash
gcloud storage buckets create "gs://$MONITORING_BUCKET" \
  --location="$REGION"
```

## Container Requirement

Cloud Run Jobs use the project container image. The image must include:

- `scripts/batch_load_inference_logs.py`
- `scripts/publish_drift_reports.py`
- Python dependencies from `pyproject.toml`

If the Dockerfile does not copy `scripts/`, add:

```dockerfile
COPY scripts ./scripts
```

Then rebuild and push the image through the CI/CD pipeline.

## Create Cloud Run Jobs

Set the image URI:

```bash
export IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE:latest"
```

Create the inference log loader job:

```bash
gcloud run jobs create nea-load-inference-logs \
  --image "$IMAGE_URI" \
  --region "$REGION" \
  --service-account "$MONITORING_SA" \
  --command python \
  --args scripts/batch_load_inference_logs.py,--project-id,"$PROJECT_ID",--bucket,"$MONITORING_BUCKET",--dataset,near_earth_monitoring,--table,inference_events,--logs,logs/*.jsonl
```

Create the drift report publisher job:

```bash
gcloud run jobs create nea-publish-drift-reports \
  --image "$IMAGE_URI" \
  --region "$REGION" \
  --service-account "$MONITORING_SA" \
  --command python \
  --args scripts/publish_drift_reports.py,--project-id,"$PROJECT_ID",--bucket,"$MONITORING_BUCKET",--dataset,near_earth_monitoring,--report,pha=reports/pha_drift.html,--report-json,pha=reports/pha_drift.json,--report,moid=reports/moid_drift.html,--report-json,moid=reports/moid_drift.json,--model-version,local,--window-start,2026-05-28T00:00:00Z,--window-end,2026-05-29T00:00:00Z
```

For existing jobs, use `gcloud run jobs update` with the same flags.

## Test Jobs Manually

Run the loader:

```bash
gcloud run jobs execute nea-load-inference-logs \
  --region "$REGION" \
  --wait
```

Run the drift publisher:

```bash
gcloud run jobs execute nea-publish-drift-reports \
  --region "$REGION" \
  --wait
```

Inspect logs:

```bash
gcloud run jobs executions list \
  --job nea-load-inference-logs \
  --region "$REGION"

gcloud logging read \
  'resource.type="cloud_run_job"' \
  --limit 50 \
  --format="value(textPayload)"
```

## Create Cloud Scheduler Triggers

Create an hourly schedule for inference loading:

```bash
gcloud scheduler jobs create http nea-load-inference-logs-hourly \
  --location "$REGION" \
  --schedule "0 * * * *" \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/nea-load-inference-logs:run" \
  --http-method POST \
  --oauth-service-account-email "$MONITORING_SA"
```

Create a daily schedule for drift publishing:

```bash
gcloud scheduler jobs create http nea-publish-drift-reports-daily \
  --location "$REGION" \
  --schedule "30 2 * * *" \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/nea-publish-drift-reports:run" \
  --http-method POST \
  --oauth-service-account-email "$MONITORING_SA"
```

Trigger a scheduled job manually:

```bash
gcloud scheduler jobs run nea-load-inference-logs-hourly \
  --location "$REGION"
```

## Expected BigQuery Tables

The loader creates or writes to:

- `near_earth_monitoring.inference_events`

The drift publisher creates or writes to:

- `near_earth_monitoring.drift_reports`
- `near_earth_monitoring.drift_metrics`

Useful checks:

```sql
SELECT model_name, status, COUNT(*) AS row_count
FROM `PROJECT_ID.near_earth_monitoring.inference_events`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY model_name, status;
```

```sql
SELECT model_name, created_at, summary
FROM `PROJECT_ID.near_earth_monitoring.drift_reports`
ORDER BY created_at DESC
LIMIT 10;
```

## Failure Handling

If a job fails:

1. Check the Cloud Run Job execution logs.
2. Verify the monitoring service account has GCS and BigQuery permissions.
3. Verify input files exist where the job expects them.
4. Re-run the job manually with `gcloud run jobs execute`.
5. If the job writes duplicate rows, use a fresh batch/report ID or deduplicate
   in downstream queries by `request_id` or `report_id`.

## Recommended Schedule

- Inference loader: hourly for active APIs, daily for low-volume demos.
- Drift publisher: daily after the inference loader has completed.

Keep the drift schedule later than the loader schedule so the current window has
already landed in BigQuery before report publishing starts.
