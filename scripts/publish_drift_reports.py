"""Publish local drift reports to GCS and BigQuery.

This script handles the "reporting" side of local Evidently monitoring:

1. Upload full report files, such as HTML/JSON, to Cloud Storage.
2. Write one summary row per report to BigQuery drift_reports.
3. Write one feature-level row per drifted/analyzed feature to BigQuery drift_metrics.

Example:
    uv run scripts/publish_drift_reports.py \
        --project-id project-3e6b348d-e2ae-4a47-9af \
        --bucket project-3e6b348d-e2ae-4a47-9af_cloudbuild \
        --report pha=reports/pha_drift.html \
        --report-json pha=reports/pha_drift.json \
        --report moid=reports/moid_drift.html \
        --report-json moid=reports/moid_drift.json \
        --model-version local \
        --window-start 2026-05-16T00:00:00Z \
        --window-end 2026-05-17T00:00:00Z

If you do not have JSON snapshots yet, omit --report-json. The script will
still upload HTML reports and write drift_reports summary rows, but it will not
write per-feature drift_metrics rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery, storage
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Missing Google Cloud libraries. Install them with:\n"
        "  pip install google-cloud-storage google-cloud-bigquery"
    ) from exc


DEFAULT_DATASET = "near_earth_monitoring"
DEFAULT_REPORTS_TABLE = "drift_reports"
DEFAULT_METRICS_TABLE = "drift_metrics"
DEFAULT_GCS_PREFIX = "monitoring/drift_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local drift reports to GCS and write drift metadata to BigQuery."
    )
    parser.add_argument("--project-id", required=True, help="GCP project ID.")
    parser.add_argument("--bucket", required=True, help="Cloud Storage bucket name, without gs://.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="BigQuery dataset name.")
    parser.add_argument("--reports-table", default=DEFAULT_REPORTS_TABLE)
    parser.add_argument("--metrics-table", default=DEFAULT_METRICS_TABLE)
    parser.add_argument(
        "--location",
        default="us-central1",
        help="BigQuery dataset/table location. Use the same region as your bucket when possible.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=DEFAULT_GCS_PREFIX,
        help="Cloud Storage prefix where report artifacts will be uploaded.",
    )
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="Local report artifact to upload. Repeat for pha/moid. Example: pha=reports/pha.html",
    )
    parser.add_argument(
        "--report-json",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Optional Evidently JSON snapshot for feature-level metrics. Example: pha=reports/pha.json",
    )
    parser.add_argument("--model-version", default="local")
    parser.add_argument("--report-type", default="data_drift")
    parser.add_argument(
        "--window-start",
        required=True,
        help="Monitoring window start timestamp, for example 2026-05-16T00:00:00Z.",
    )
    parser.add_argument(
        "--window-end",
        required=True,
        help="Monitoring window end timestamp, for example 2026-05-17T00:00:00Z.",
    )
    parser.add_argument(
        "--create-resources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create BigQuery dataset/tables if they do not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print rows, but do not upload to GCS or write BigQuery.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def drift_reports_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("report_id", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("window_start", "TIMESTAMP"),
        bigquery.SchemaField("window_end", "TIMESTAMP"),
        bigquery.SchemaField("report_type", "STRING"),
        bigquery.SchemaField("report_uri", "STRING"),
        bigquery.SchemaField("json_report_uri", "STRING"),
        bigquery.SchemaField("status", "STRING"),
        bigquery.SchemaField("summary", "JSON"),
    ]


def drift_metrics_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("report_id", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
        bigquery.SchemaField("model_name", "STRING"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("metric_name", "STRING"),
        bigquery.SchemaField("feature_name", "STRING"),
        bigquery.SchemaField("drift_score", "FLOAT"),
        bigquery.SchemaField("threshold", "FLOAT"),
        bigquery.SchemaField("drift_detected", "BOOL"),
        bigquery.SchemaField("p_value", "FLOAT"),
        bigquery.SchemaField("details", "JSON"),
    ]


def parse_model_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected MODEL=PATH, got {value!r}")
    model_name, path = value.split("=", 1)
    model_name = model_name.strip()
    if not model_name:
        raise argparse.ArgumentTypeError(f"Missing model name in {value!r}")
    local_path = Path(path).expanduser().resolve()
    if not local_path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {local_path}")
    return model_name, local_path


def parse_model_path_args(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        model_name, local_path = parse_model_path(value)
        parsed[model_name] = local_path
    return parsed


def normalize_timestamp(value: str) -> str:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()


def ensure_dataset(client: bigquery.Client, project_id: str, dataset_name: str, location: str) -> None:
    dataset_id = f"{project_id}.{dataset_name}"
    try:
        client.get_dataset(dataset_id)
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = location
        client.create_dataset(dataset)
        logging.info("Created BigQuery dataset %s", dataset_id)


def ensure_table(
    client: bigquery.Client,
    project_id: str,
    dataset_name: str,
    table_name: str,
    schema: list[bigquery.SchemaField],
    partition_field: str,
    clustering_fields: list[str],
) -> None:
    table_id = f"{project_id}.{dataset_name}.{table_name}"
    try:
        client.get_table(table_id)
        return
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
        table.clustering_fields = clustering_fields
        client.create_table(table)
        logging.info("Created BigQuery table %s", table_id)


def upload_file_to_gcs(
    storage_client: storage.Client,
    bucket_name: str,
    local_path: Path,
    gcs_prefix: str,
    model_name: str,
    report_id: str,
) -> str:
    content_type, _ = mimetypes.guess_type(local_path.name)
    blob_name = f"{gcs_prefix.rstrip('/')}/{model_name}/{report_id}/{local_path.name}"
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(local_path, content_type=content_type)
    print(f"gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def walk_json(value: Any) -> list[dict[str, Any]]:
    found = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(walk_json(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_json(child))
    return found


def first_present(mapping: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1", "detected"}:
            return True
        if lowered in {"false", "no", "0", "not_detected"}:
            return False
    return None


def feature_name_from_metric(metric: dict[str, Any]) -> str | None:
    return first_present(
        metric,
        [
            "feature_name",
            "column_name",
            "column",
            "name",
            "field_name",
            "target_name",
        ],
    )


def extract_feature_metrics(report_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract feature drift rows from common Evidently JSON/dict shapes.

    Evidently has changed its serialized report structure across versions. This
    function intentionally uses a tolerant search for dictionaries that look
    like per-feature drift metrics.
    """
    metrics = []

    for item in walk_json(report_json):
        feature_name = feature_name_from_metric(item)
        drift_score = first_present(
            item,
            ["drift_score", "stattest_value", "statistic", "score", "value"],
        )
        threshold = first_present(item, ["threshold", "stattest_threshold"])
        drift_detected = first_present(
            item,
            ["drift_detected", "detected", "is_drifted", "drifted"],
        )
        p_value = first_present(item, ["p_value", "pvalue"])

        has_drift_signal = any(
            key in item
            for key in [
                "drift_score",
                "stattest_value",
                "drift_detected",
                "is_drifted",
                "p_value",
                "pvalue",
            ]
        )
        if not feature_name or not has_drift_signal:
            continue

        metrics.append(
            {
                "metric_name": str(first_present(item, ["metric_name", "metric"]) or "data_drift"),
                "feature_name": str(feature_name),
                "drift_score": to_float(drift_score),
                "threshold": to_float(threshold),
                "drift_detected": to_bool(drift_detected),
                "p_value": to_float(p_value),
                "details": item,
            }
        )

    deduped = {}
    for metric in metrics:
        key = (
            metric["metric_name"],
            metric["feature_name"],
            metric["drift_score"],
            metric["threshold"],
            metric["drift_detected"],
            metric["p_value"],
        )
        deduped[key] = metric
    return list(deduped.values())


def summarize_metrics(feature_metrics: list[dict[str, Any]], report_json: dict[str, Any] | None) -> dict[str, Any]:
    drift_flags = [metric["drift_detected"] for metric in feature_metrics]
    detected_flags = [flag for flag in drift_flags if flag is not None]
    return {
        "feature_metric_count": len(feature_metrics),
        "drifted_feature_count": sum(1 for flag in detected_flags if flag),
        "drift_detected": any(detected_flags) if detected_flags else None,
        "json_report_available": report_json is not None,
    }


def insert_rows(client: bigquery.Client, table_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        logging.info("No rows to insert into %s", table_id)
        return

    table = client.get_table(table_id)
    json_fields = {field.name for field in table.schema if field.field_type.upper() == "JSON"}
    prepared_rows = []
    for row in rows:
        prepared_row = dict(row)
        for field_name in json_fields:
            field_value = prepared_row.get(field_name)
            if field_value is not None and not isinstance(field_value, str):
                prepared_row[field_name] = json.dumps(field_value)
        prepared_rows.append(prepared_row)

    try:
        errors = client.insert_rows_json(table_id, prepared_rows)
    except Exception as errors:
        raise RuntimeError(f"BigQuery insert failed for {table_id}: {errors}") from errors
    if errors:
        raise RuntimeError(f"BigQuery insert failed for {table_id}: {errors}")


def build_report_id(model_name: str, window_start: str, window_end: str) -> str:
    start = window_start.replace(":", "").replace("-", "").replace("+0000", "Z")
    end = window_end.replace(":", "").replace("-", "").replace("+0000", "Z")
    return f"{model_name}-{start}-{end}-{uuid.uuid4().hex[:8]}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    reports = parse_model_path_args(args.report)
    json_reports = parse_model_path_args(args.report_json)
    window_start = normalize_timestamp(args.window_start)
    window_end = normalize_timestamp(args.window_end)
    created_at = datetime.now(timezone.utc).isoformat()

    bq_client = bigquery.Client(project=args.project_id)
    storage_client = storage.Client(project=args.project_id)

    if args.create_resources and not args.dry_run:
        ensure_dataset(bq_client, args.project_id, args.dataset, args.location)
        ensure_table(
            bq_client,
            args.project_id,
            args.dataset,
            args.reports_table,
            drift_reports_schema(),
            partition_field="created_at",
            clustering_fields=["model_name", "report_type"],
        )
        ensure_table(
            bq_client,
            args.project_id,
            args.dataset,
            args.metrics_table,
            drift_metrics_schema(),
            partition_field="created_at",
            clustering_fields=["model_name", "feature_name"],
        )

    drift_report_rows = []
    drift_metric_rows = []

    for model_name, report_path in reports.items():
        report_id = build_report_id(model_name, window_start, window_end)
        report_json_path = json_reports.get(model_name)
        report_json = load_json(report_json_path) if report_json_path else None
        print(f"REPORT_JSON:  {report_json if report_json == None else 1}")
        
        feature_metrics = extract_feature_metrics(report_json) if report_json else []
        summary = summarize_metrics(feature_metrics, report_json)

        if args.dry_run:
            report_uri = f"dry-run://{report_path}"
            json_report_uri = f"dry-run://{report_json_path}" if report_json_path else None
        else:
            report_uri = upload_file_to_gcs(
                storage_client,
                args.bucket,
                report_path,
                args.gcs_prefix,
                model_name,
                report_id,
            )
            json_report_uri = None
            if report_json_path:
                json_report_uri = upload_file_to_gcs(
                    storage_client,
                    args.bucket,
                    report_json_path,
                    args.gcs_prefix,
                    model_name,
                    report_id,
                )

        drift_report_rows.append(
            {
                "report_id": report_id,
                "created_at": created_at,
                "model_name": model_name,
                "model_version": args.model_version,
                "window_start": window_start,
                "window_end": window_end,
                "report_type": args.report_type,
                "report_uri": report_uri,
                "json_report_uri": json_report_uri,
                "status": "success",
                "summary": summary,
            }
        )

        for metric in feature_metrics:
            drift_metric_rows.append(
                {
                    "report_id": report_id,
                    "created_at": created_at,
                    "model_name": model_name,
                    "model_version": args.model_version,
                    **metric,
                }
            )

        logging.info(
            "Prepared %s report row and %s feature metric rows for %s",
            1,
            len(feature_metrics),
            model_name,
        )

    if args.dry_run:
        logging.info("Dry run report rows: %s", json.dumps(drift_report_rows, indent=2, default=str))
        logging.info("Dry run metric row count: %s", len(drift_metric_rows))
        return

    reports_table_id = f"{args.project_id}.{args.dataset}.{args.reports_table}"
    metrics_table_id = f"{args.project_id}.{args.dataset}.{args.metrics_table}"
    insert_rows(bq_client, reports_table_id, drift_report_rows)
    insert_rows(bq_client, metrics_table_id, drift_metric_rows)

    logging.info("Inserted %s rows into %s", len(drift_report_rows), reports_table_id)
    logging.info("Inserted %s rows into %s", len(drift_metric_rows), metrics_table_id)


if __name__ == "__main__":
    main()
