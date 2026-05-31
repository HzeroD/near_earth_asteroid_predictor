"""Focused tests for API inference logging and monitoring loader helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.batch_load_inference_logs as inference_loader
import scripts.publish_drift_reports as drift_publisher
from near_earth_asteroid_predictor import api


VALID_PHA_FEATURES = {
    "H": 13.82,
    "diameter_km": 4.2,
    "size_category": "Large",
    "class_code": "AMO",
    "eccentricity": 0.5712,
    "semimajor_axis_au": 2.474,
    "inclination_deg": 9.4,
    "perihelion_distance_au": 1.061,
    "aphelion_distance_au": 3.89,
    "orbital_period_days": 1420.0,
    "moid_au": 0.0717,
    "mean_motion_deg_day": 0.2533,
    "condition_code": 0,
    "data_arc": 39281.0,
}


class FakeTransformer:
    def transform(self, _df):
        return FakeTransformedFeatures([[1.0, 2.0]])

    def get_feature_names_out(self):
        return ["feature_a", "feature_b"]


class FakeTransformedFeatures:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _df):
        return self.prediction


class FakeSchemaField:
    def __init__(self, name: str, field_type: str):
        self.name = name
        self.field_type = field_type


class FakeBigQueryClient:
    def __init__(self, schema, insert_errors=None):
        self.schema = schema
        self.insert_errors = insert_errors or []
        self.inserted_rows = None

    def get_table(self, _table_id):
        return type("FakeTable", (), {"schema": self.schema})()

    def insert_rows_json(self, _table_id, rows):
        self.inserted_rows = rows
        return self.insert_errors


class FakeBlob:
    def __init__(self):
        self.uploaded_filename = None
        self.content_type = None

    def upload_from_filename(self, filename, content_type=None):
        self.uploaded_filename = filename
        self.content_type = content_type


class FakeBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        self.blobs[name] = FakeBlob()
        return self.blobs[name]


class FakeStorageClient:
    def __init__(self):
        self.buckets = {}

    def bucket(self, name):
        self.buckets[name] = FakeBucket()
        return self.buckets[name]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_inference_event_routes_events_to_model_specific_jsonl(tmp_path, monkeypatch):
    log_paths = {
        "pha": tmp_path / "pha.jsonl",
        "moid": tmp_path / "moid.jsonl",
    }
    monkeypatch.setattr(api, "INFERENCE_LOG_PATHS", log_paths)

    api.log_inference_event(
        {
            "timestamp": "2026-05-29T12:00:00+00:00",
            "model_name": "pha",
            "features": {"H": 13.82},
            "prediction": [1],
            "status": "success",
            "latency_ms": 12.5,
            "error": None,
        }
    )

    rows = read_jsonl(log_paths["pha"])
    assert rows[0]["model_name"] == "pha"
    assert rows[0]["prediction"] == [1]
    assert not log_paths["moid"].exists()


def test_predict_pha_writes_successful_inference_event(tmp_path, monkeypatch):
    log_path = tmp_path / "inference_events.jsonl"
    monkeypatch.setattr(api, "INFERENCE_LOG_PATHS", {"pha": log_path})
    monkeypatch.setattr(api, "columntransformer_pha", FakeTransformer())
    monkeypatch.setattr(api, "model_pha", FakeModel([1]))

    response = TestClient(api.app).post("/predict_pha", json=VALID_PHA_FEATURES)

    assert response.status_code == 200
    assert response.json() == {"pha_prediction": [1]}
    event = read_jsonl(log_path)[0]
    assert event["model_name"] == "pha"
    assert event["status"] == "success"
    assert event["features"]["H"] == VALID_PHA_FEATURES["H"]
    assert event["prediction"] == [1]
    assert event["error"] is None
    assert isinstance(event["latency_ms"], int | float)


def test_normalize_logs_writes_bigquery_ready_rows(tmp_path):
    source_log = tmp_path / "inference.jsonl"
    output_log = tmp_path / "normalized.jsonl"
    source_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-29T12:00:00+00:00",
                        "model_name": "pha",
                        "model_version": "local",
                        "features": {"H": 13.82},
                        "prediction": [1],
                        "status": "success",
                        "latency_ms": 3.4,
                        "error": None,
                    }
                ),
                "",
                json.dumps(
                    {
                        "request_id": "existing-id",
                        "timezone": "2026-05-29T12:01:00+00:00",
                        "model_name": "moid",
                        "prediction": [0.12],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    row_count = inference_loader.normalize_logs([source_log], output_log, "batch-123")

    rows = read_jsonl(output_log)
    assert row_count == 2
    assert rows[0]["endpoint"] == "/predict_pha"
    assert rows[0]["source_line"] == 1
    assert rows[0]["load_batch_id"] == "batch-123"
    assert rows[0]["request_id"]
    assert rows[1]["request_id"] == "existing-id"
    assert rows[1]["timestamp"] == "2026-05-29T12:01:00+00:00"


def test_normalize_logs_rejects_invalid_json(tmp_path):
    source_log = tmp_path / "bad.jsonl"
    source_log.write_text("{not json}", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        inference_loader.normalize_logs([source_log], tmp_path / "out.jsonl", "batch-123")


def test_upload_to_gcs_uses_expected_batch_path(tmp_path):
    source_path = tmp_path / "batch.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    storage_client = FakeStorageClient()

    uri = inference_loader.upload_to_gcs(
        storage_client=storage_client,
        bucket_name="monitoring-bucket",
        source_path=source_path,
        gcs_prefix="inference_logs/batches/",
        load_batch_id="batch-123",
    )

    bucket = storage_client.buckets["monitoring-bucket"]
    blob = bucket.blobs["inference_logs/batches/batch-123.jsonl"]
    assert uri == "gs://monitoring-bucket/inference_logs/batches/batch-123.jsonl"
    assert blob.uploaded_filename == source_path


def test_extract_feature_metrics_and_summary_from_drift_report_json():
    report_json = {
        "metrics": [
            {
                "result": {
                    "drift_by_columns": {
                        "H": {
                            "column_name": "H",
                            "stattest_name": "Wasserstein distance",
                            "stattest_threshold": 0.1,
                            "drift_score": "0.25",
                            "drift_detected": "true",
                        },
                        "diameter_km": {
                            "column_name": "diameter_km",
                            "stattest_threshold": 0.1,
                            "drift_score": 0.03,
                            "drift_detected": False,
                        },
                    }
                }
            }
        ]
    }

    metrics = drift_publisher.extract_feature_metrics(report_json)
    summary = drift_publisher.summarize_metrics(metrics, report_json)

    metrics_by_feature = {metric["feature_name"]: metric for metric in metrics}
    assert metrics_by_feature["H"]["drift_score"] == 0.25
    assert metrics_by_feature["H"]["drift_detected"] is True
    assert metrics_by_feature["diameter_km"]["drift_detected"] is False
    assert summary == {
        "feature_metric_count": 2,
        "drifted_feature_count": 1,
        "drift_detected": True,
        "json_report_available": True,
    }


def test_insert_rows_serializes_bigquery_json_fields():
    client = FakeBigQueryClient(
        schema=[
            FakeSchemaField("summary", "JSON"),
            FakeSchemaField("model_name", "STRING"),
        ]
    )

    drift_publisher.insert_rows(
        client,
        "project.dataset.drift_reports",
        [{"summary": {"drift_detected": True}, "model_name": "pha"}],
    )

    assert client.inserted_rows == [
        {"summary": '{"drift_detected": true}', "model_name": "pha"}
    ]


def test_upload_file_to_gcs_builds_model_scoped_report_uri(tmp_path):
    report_path = tmp_path / "report.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    storage_client = FakeStorageClient()

    uri = drift_publisher.upload_file_to_gcs(
        storage_client=storage_client,
        bucket_name="monitoring-bucket",
        local_path=report_path,
        gcs_prefix="monitoring/drift_reports/",
        model_name="pha",
        report_id="report-123",
    )

    bucket = storage_client.buckets["monitoring-bucket"]
    blob = bucket.blobs["monitoring/drift_reports/pha/report-123/report.html"]
    assert uri == "gs://monitoring-bucket/monitoring/drift_reports/pha/report-123/report.html"
    assert blob.uploaded_filename == report_path
    assert blob.content_type == "text/html"
