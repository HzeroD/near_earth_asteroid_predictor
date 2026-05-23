"""Batch load inference JSONL logs into BigQuery via Cloud Storage.

This script is intended for monitoring pipelines where inference should stay
fast and storage failures should be handled outside the prediction request.

Example:
    python scripts/batch_load_inference_logs.py \
        --project-id my-gcp-project \
        --bucket my-nea-monitoring-bucket \
        --dataset near_earth_monitoring \
        --table inference_events \
        --logs src/near_earth_asteroid_predictor/logs/*.jsonl

Prerequisites:
    pip install google-cloud-storage google-cloud-bigquery
    gcloud auth application-default login
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google.api_core.exceptions import BadRequest
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery, storage
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Missing Google Cloud libraries. Install them with:\n"
        "  pip install google-cloud-storage google-cloud-bigquery"
    ) from exc


DEFAULT_DATASET = "near_earth_monitoring"
DEFAULT_TABLE = "inference_events"
DEFAULT_GCS_PREFIX = "inference_logs/batches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize local inference JSONL files, upload them to GCS, and load them into BigQuery."
    )
    parser.add_argument("--project-id", required=True, help="GCP project ID.")
    parser.add_argument("--bucket", required=True, help="Cloud Storage bucket name, without gs://.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="BigQuery dataset name.")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="BigQuery table name.")
    parser.add_argument(
        "--location",
        default="us-central1",
        help="BigQuery dataset/load job location. Use the same region as your bucket when possible.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=DEFAULT_GCS_PREFIX,
        help="Cloud Storage prefix for normalized batch files.",
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more local JSONL files. Shell globs are supported by your shell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and normalize logs, but do not upload to GCS or load BigQuery.",
    )
    parser.add_argument(
        "--create-resources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the BigQuery dataset/table if they do not exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def table_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("request_id", "STRING"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("endpoint", "STRING"),
        bigquery.SchemaField("features", "JSON"),
        bigquery.SchemaField("prediction", "JSON"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("latency_ms", "FLOAT"),
        bigquery.SchemaField("error", "STRING"),
        bigquery.SchemaField("source_file", "STRING"),
        bigquery.SchemaField("source_line", "INTEGER"),
        bigquery.SchemaField("load_batch_id", "STRING"),
        bigquery.SchemaField("loaded_at", "TIMESTAMP"),
    ]


def infer_endpoint(model_name: str | None) -> str | None:
    if not model_name:
        return None
    endpoint_by_model = {
        "pha": "/predict_pha",
        "moid": "/predict_moid",
        "magnitude": "/predict_magnitude",
        "mag": "/predict_magnitude",
        "absolute_magnitude": "/predict_magnitude",
    }
    return endpoint_by_model.get(model_name, f"/predict_{model_name}")


def stable_request_id(source_file: Path, line_number: int, event: dict[str, Any]) -> str:
    if event.get("request_id"):
        return str(event["request_id"])

    raw_key = json.dumps(
        {
            "source_file": str(source_file),
            "source_line": line_number,
            "timestamp": event.get("timestamp"),
            "model_name": event.get("model_name"),
            "features": event.get("features"),
            "prediction": event.get("prediction"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def normalize_event(
    event: dict[str, Any],
    source_file: Path,
    line_number: int,
    load_batch_id: str,
    loaded_at: str,
) -> dict[str, Any]:
    model_name = event.get("model_name")
    timestamp = event.get("timestamp") or event.get("timezone")

    return {
        "request_id": stable_request_id(source_file, line_number, event),
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": event.get("model_version"),
        "endpoint": infer_endpoint(model_name),
        "features": event.get("features"),
        "prediction": event.get("prediction"),
        "status": event.get("status"),
        "latency_ms": event.get("latency_ms"),
        "error": event.get("error"),
        "source_file": str(source_file),
        "source_line": line_number,
        "load_batch_id": load_batch_id,
        "loaded_at": loaded_at,
    }


def normalize_logs(log_paths: list[Path], output_path: Path, load_batch_id: str) -> int:
    loaded_at = datetime.now(timezone.utc).isoformat()
    row_count = 0

    with output_path.open("w", encoding="utf-8") as output:
        for log_path in log_paths:
            if not log_path.exists():
                raise FileNotFoundError(f"Log file not found: {log_path}")

            with log_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {log_path} at line {line_number}: {exc}"
                        ) from exc

                    if not isinstance(event, dict):
                        raise ValueError(f"Expected JSON object in {log_path} at line {line_number}")

                    normalized = normalize_event(
                        event=event,
                        source_file=log_path,
                        line_number=line_number,
                        load_batch_id=load_batch_id,
                        loaded_at=loaded_at,
                    )
                    output.write(json.dumps(normalized, default=str) + "\n")
                    row_count += 1

    return row_count


def ensure_dataset_and_table(
    client: bigquery.Client,
    project_id: str,
    dataset_name: str,
    table_name: str,
    location: str,
) -> bigquery.Table:
    dataset_id = f"{project_id}.{dataset_name}"
    table_id = f"{dataset_id}.{table_name}"

    try:
        client.get_dataset(dataset_id)
        logging.info("Found BigQuery dataset %s", dataset_id)
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = location
        client.create_dataset(dataset)
        logging.info("Created BigQuery dataset %s", dataset_id)

    try:
        table = client.get_table(table_id)
        validate_table_schema(table)
        return table
    except NotFound:
        table = bigquery.Table(table_id, schema=table_schema())
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp",
        )
        table.clustering_fields = ["model_name", "status"]
        created_table = client.create_table(table)
        logging.info("Created BigQuery table %s", table_id)
        return created_table


def validate_table_schema(table: bigquery.Table) -> None:
    expected_types = {field.name: field.field_type for field in table_schema()}
    actual_types = {field.name: field.field_type for field in table.schema}
    mismatches = []

    for name, expected_type in expected_types.items():
        actual_type = actual_types.get(name)
        if actual_type is None:
            mismatches.append(f"{name}: missing, expected {expected_type}")
        elif actual_type != expected_type:
            mismatches.append(f"{name}: found {actual_type}, expected {expected_type}")

    if mismatches:
        mismatch_text = "\n  - ".join(mismatches)
        raise SystemExit(
            f"BigQuery table {table.full_table_id} does not match this loader's schema.\n"
            f"  - {mismatch_text}\n\n"
            "The most likely issue is that the table was created with prediction as BOOL.\n"
            "Use a fresh table name, for example:\n"
            "  --table inference_events_v2\n\n"
            "Or delete/recreate the existing empty table in BigQuery with prediction as JSON."
        )


def upload_to_gcs(
    storage_client: storage.Client,
    bucket_name: str,
    source_path: Path,
    gcs_prefix: str,
    load_batch_id: str,
) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob_name = f"{gcs_prefix.rstrip('/')}/{load_batch_id}.jsonl"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_path)
    return f"gs://{bucket_name}/{blob_name}"


def load_gcs_jsonl_to_bigquery(
    client: bigquery.Client,
    table_id: str,
    source_uri: str,
    location: str,
) -> bigquery.LoadJob:
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=table_schema(),
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ignore_unknown_values=False,
        max_bad_records=0,
    )
    load_job = client.load_table_from_uri(
        source_uri,
        table_id,
        job_config=job_config,
        location=location,
    )
    try:
        load_job.result()
    except BadRequest as exc:
        if load_job.errors:
            logging.error("BigQuery load job errors: %s", json.dumps(load_job.errors, indent=2))
        raise exc
    return load_job


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log_paths = [Path(path).resolve() for path in args.logs]
    logging.info("Preparing to load log files: %s", log_paths)
    load_batch_id = uuid.uuid4().hex

    with tempfile.TemporaryDirectory() as temp_dir:
        normalized_path = Path(temp_dir) / f"{load_batch_id}.jsonl"
        row_count = normalize_logs(log_paths, normalized_path, load_batch_id)

        logging.info("Normalized %s rows into %s", row_count, normalized_path)

        if row_count == 0:
            logging.warning("No rows found. Nothing to load.")
            return

        if args.dry_run:
            logging.info("Dry run complete. Skipping GCS upload and BigQuery load.")
            return

        bq_client = bigquery.Client(project=args.project_id)
        storage_client = storage.Client(project=args.project_id)

        table_id = f"{args.project_id}.{args.dataset}.{args.table}"
        if args.create_resources:
            ensure_dataset_and_table(
                client=bq_client,
                project_id=args.project_id,
                dataset_name=args.dataset,
                table_name=args.table,
                location=args.location,
            )

        source_uri = upload_to_gcs(
            storage_client=storage_client,
            bucket_name=args.bucket,
            source_path=normalized_path,
            gcs_prefix=args.gcs_prefix,
            load_batch_id=load_batch_id,
        )
        logging.info("Uploaded normalized batch to %s", source_uri)

        load_job = load_gcs_jsonl_to_bigquery(
            client=bq_client,
            table_id=table_id,
            source_uri=source_uri,
            location=args.location,
        )
        logging.info(
            "Loaded %s rows into %s with job %s",
            load_job.output_rows,
            table_id,
            load_job.job_id,
        )


if __name__ == "__main__":
    main()
