"""
Training pipeline derived from notebooks/train.ipynb.

The notebook trains two production artifacts:
1. PHA classification with a balanced RandomForestClassifier.
2. MOID regression with a RandomForestRegressor.

It also saves the fitted column transformers and reference train/test sets used
by the API and monitoring jobs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "near_earth_asteroids_2025.csv"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts"
ARTIFACT_DATA_PATH = ARTIFACT_PATH / "data"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

GCS_BUCKET_NAME = "project-3e6b348d-e2ae-4a47-9af_cloudbuild"
GCS_ARTIFACT_PREFIX = "artifacts/"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "train.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


DROP_FOR_PHA = [
    "spkid",
    "full_name",
    "pdes",
    "name",
    "diameter_m",
    "moid_lunar_distances",
    "albedo",
    "rot_per_h",
    "moid_km",
    "per_y",
    "data_arc_years",
    "diameter_is_estimated",
    "pha",
    "first_obs",
    "last_obs",
]

DROP_FOR_MOID = [
    "spkid",
    "full_name",
    "pdes",
    "name",
    "diameter_m",
    "moid_lunar_distances",
    "albedo",
    "rot_per_h",
    "moid_km",
    "moid_au",
    "per_y",
    "data_arc_years",
    "diameter_is_estimated",
    "first_obs",
    "last_obs",
]


def load_near_earth_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the NEA dataset and apply the notebook's column renames."""
    logger.info("Loading NEA data from %s", data_path)
    df_near = pd.read_csv(data_path)

    df_near.rename(
        {
            "q": "perihelion_distance_au",
            "ad": "aphelion_distance_au",
            "e": "eccentricity",
            "a": "semimajor_axis_au",
            "i": "inclination_deg",
            "per": "orbital_period_days",
            "n": "mean_motion_deg_day",
            "rot_per": "rot_per_h",
            "class": "class_code",
        },
        axis=1,
        inplace=True,
    )

    logger.info("Loaded %s records", len(df_near))
    return df_near


def clean_size_category(series: pd.Series) -> pd.Series:
    """Match the notebook's size-category cleanup while tolerating missing text."""
    return series.apply(
        lambda value: value.split(" ", 1)[0].strip()
        if isinstance(value, str) and " " in value
        else value
    )


def prepare_training_datasets(
    df_near: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create PHA and MOID feature matrices exactly as in the notebook."""
    X_near_pha = df_near.drop(DROP_FOR_PHA, axis=1).copy()
    X_near_pha["size_category"] = clean_size_category(X_near_pha["size_category"])
    y_pha = df_near["pha"]

    X_near_moid = df_near.drop(DROP_FOR_MOID, axis=1).copy()
    X_near_moid["size_category"] = clean_size_category(X_near_moid["size_category"])
    y_moid = df_near["moid_au"]

    logger.info("PHA dataset shape: %s", X_near_pha.shape)
    logger.info("MOID dataset shape: %s", X_near_moid.shape)
    logger.info("PHA positive class rate: %.2f%%", y_pha.mean() * 100)

    return X_near_pha, y_pha, X_near_moid, y_moid


def create_preprocessor() -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("simple_imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("standard_scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical_transformer",
                categorical_transformer,
                make_column_selector(dtype_include=["object", "category", "str"]),
            ),
            (
                "numeric_transformer",
                numeric_transformer,
                make_column_selector(dtype_include="number"),
            ),
        ]
    )


def create_model_pipelines() -> tuple[dict[str, Pipeline], dict[str, Pipeline]]:
    """Create the notebook's candidate model pipelines."""
    pipelines_classification = {
        "Random Forest Classifier": make_pipeline(
            create_preprocessor(),
            RandomForestClassifier(class_weight="balanced"),
        ),
        "Logistic Regression": make_pipeline(
            create_preprocessor(),
            LogisticRegression(class_weight="balanced"),
        ),
        "Gradient Boosting Classifier": make_pipeline(
            create_preprocessor(),
            GradientBoostingClassifier(),
        ),
    }

    pipelines_regression = {
        "Random Forest Regressor": make_pipeline(
            create_preprocessor(),
            RandomForestRegressor(),
        ),
        "Decision Tree Regressor": make_pipeline(
            create_preprocessor(),
            DecisionTreeRegressor(),
        ),
        "Gradient Boosting Regressor": make_pipeline(
            create_preprocessor(),
            GradientBoostingRegressor(),
        ),
        "Linear Regression": make_pipeline(
            create_preprocessor(),
            LinearRegression(),
        ),
    }

    return pipelines_classification, pipelines_regression


def split_training_data(
    X_near_pha: pd.DataFrame,
    y_pha: pd.Series,
    X_near_moid: pd.DataFrame,
    y_moid: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train_pha, X_test_pha, y_train_pha, y_test_pha = train_test_split(
        X_near_pha,
        y_pha,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train_moid, X_test_moid, y_train_moid, y_test_moid = train_test_split(
        X_near_moid,
        y_moid,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return (
        X_train_pha,
        X_test_pha,
        y_train_pha,
        y_test_pha,
        X_train_moid,
        X_test_moid,
        y_train_moid,
        y_test_moid,
    )


def drop_missing_targets(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_target_idx = y[y.isna()].index
    if missing_target_idx.empty:
        return X.copy(), y.copy()

    return X.drop(missing_target_idx, axis=0).copy(), y.drop(missing_target_idx).copy()


def cross_validate_all(
    X_train_pha: pd.DataFrame,
    y_train_pha: pd.Series,
    X_train_moid: pd.DataFrame,
    y_train_moid: pd.Series,
    pipelines_classification: dict[str, Pipeline],
    pipelines_regression: dict[str, Pipeline],
) -> dict[str, dict[str, float]]:
    """Run the notebook's cross-validation checks."""
    results: dict[str, dict[str, float]] = {}
    stratified_kf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    logger.info("PHA train data cross-validation:")
    X_pha_clean, y_pha_clean = drop_missing_targets(X_train_pha, y_train_pha)
    for name, pipeline in pipelines_classification.items():
        scores = cross_val_score(
            pipeline,
            X_pha_clean,
            y_pha_clean,
            cv=stratified_kf,
            scoring="roc_auc",
        )
        results[name] = {"cv_auc": scores.mean(), "cv_auc_std": scores.std()}
        logger.info("%-30s ROC-AUC = %.3f +/- %.3f", name, scores.mean(), scores.std())

    logger.info("MOID train data cross-validation:")
    X_moid_clean, y_moid_clean = drop_missing_targets(X_train_moid, y_train_moid)
    for name, pipeline in pipelines_regression.items():
        scores = cross_val_score(pipeline, X_moid_clean, y_moid_clean, cv=CV_FOLDS)
        results[name] = {"cv_r2": scores.mean(), "cv_r2_std": scores.std()}
        logger.info("%-30s R2 = %.3f +/- %.3f", name, scores.mean(), scores.std())

    return results


def save_pha_training_stats(X_train_pha: pd.DataFrame) -> None:
    """Save the CSV monitoring stats created by the notebook."""
    ARTIFACT_DATA_PATH.mkdir(parents=True, exist_ok=True)

    numeric_columns = X_train_pha.select_dtypes(include="number")
    quantiles_x_train = X_train_pha.quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9],
        axis=0,
        numeric_only=True,
    )

    X_train_stats = pd.concat(
        [
            pd.DataFrame(
                [numeric_columns.std().values.tolist()],
                columns=numeric_columns.std().index,
                index=["std"],
            ),
            quantiles_x_train,
            pd.DataFrame(
                [numeric_columns.min().values.tolist()],
                columns=numeric_columns.columns,
                index=["min"],
            ),
            pd.DataFrame(
                [numeric_columns.max().values.tolist()],
                columns=numeric_columns.columns,
                index=["max"],
            ),
        ]
    )
    X_train_stats.round(3).to_csv(ARTIFACT_DATA_PATH / "X_train_stats.csv")
    logger.info("Saved %s", ARTIFACT_DATA_PATH / "X_train_stats.csv")


def train_pha_model(
    X_train_pha: pd.DataFrame,
    y_train_pha: pd.Series,
    X_test_pha: pd.DataFrame,
    y_test_pha: pd.Series,
    pipelines_classification: dict[str, Pipeline],
) -> Pipeline:
    """Train and evaluate the notebook-selected PHA model."""
    best_model_pha = pipelines_classification["Random Forest Classifier"]
    best_model_pha.fit(X_train_pha, y_train_pha)

    y_train_pred = best_model_pha.predict(X_train_pha)
    y_test_pred = best_model_pha.predict(X_test_pha)
    y_test_proba = best_model_pha.predict_proba(X_test_pha)[:, 1]

    logger.info("PHA train F1 score: %.3f", f1_score(y_train_pha, y_train_pred))
    logger.info("PHA true positive rate in test set: %.3f", y_test_pha.mean())
    logger.info("PHA predicted positive rate: %.3f", y_test_pred.mean())
    logger.info("PHA classification report:\n%s", classification_report(y_test_pha, y_test_pred))
    logger.info("PHA test ROC-AUC score: %.3f", roc_auc_score(y_test_pha, y_test_proba))

    return best_model_pha


def train_moid_model(
    X_train_moid: pd.DataFrame,
    y_train_moid: pd.Series,
    X_test_moid: pd.DataFrame,
    y_test_moid: pd.Series,
    pipelines_regression: dict[str, Pipeline],
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Train the notebook-selected MOID model after dropping null targets."""
    X_train_clean, y_train_clean = drop_missing_targets(X_train_moid, y_train_moid)
    best_model_moid = pipelines_regression["Random Forest Regressor"]
    best_model_moid.fit(X_train_clean, y_train_clean)

    X_test_clean, y_test_clean = drop_missing_targets(X_test_moid, y_test_moid)
    if not y_test_clean.empty:
        y_pred = best_model_moid.predict(X_test_clean)
        logger.info("MOID test R2 score: %.3f", r2_score(y_test_clean, y_pred))

    return best_model_moid, X_train_clean, y_train_clean


def save_model_artifacts(
    model_pha: Pipeline,
    model_moid: Pipeline,
    X_train_pha: pd.DataFrame,
    y_train_pha: pd.Series,
    X_test_pha: pd.DataFrame,
    y_test_pha: pd.Series,
    X_train_moid: pd.DataFrame,
    y_train_moid: pd.Series,
    X_test_moid: pd.DataFrame,
    y_test_moid: pd.Series,
) -> None:
    """Save the artifacts produced by the notebook."""
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pha["randomforestclassifier"], ARTIFACT_PATH / "best_model_pha.pkl")
    joblib.dump(model_pha["columntransformer"], ARTIFACT_PATH / "columntransformer_pha.pkl")

    joblib.dump(model_moid["randomforestregressor"], ARTIFACT_PATH / "best_model_moid.pkl")
    joblib.dump(model_moid["columntransformer"], ARTIFACT_PATH / "columntransformer_moid.pkl")

    joblib.dump(X_train_pha, ARTIFACT_PATH / "X_train_pha.pkl")
    joblib.dump(y_train_pha, ARTIFACT_PATH / "y_train_pha.pkl")
    joblib.dump(X_test_pha, ARTIFACT_PATH / "X_test_pha.pkl")
    joblib.dump(y_test_pha, ARTIFACT_PATH / "y_test_pha.pkl")

    joblib.dump(X_train_moid, ARTIFACT_PATH / "X_train_moid.pkl")
    joblib.dump(y_train_moid, ARTIFACT_PATH / "y_train_moid.pkl")
    joblib.dump(X_test_moid, ARTIFACT_PATH / "X_test_moid.pkl")
    joblib.dump(y_test_moid, ARTIFACT_PATH / "y_test_moid.pkl")

    joblib.dump(X_train_moid.describe(), ARTIFACT_PATH / "X_train_moid_stats.pkl")
    logger.info("Saved model, transformer, and monitoring artifacts to %s", ARTIFACT_PATH)


def upload_local_directory(
    bucket_name: str,
    local_folder_path: Path,
    gcs_folder_path: str | None = None,
) -> None:
    """Upload a local artifact folder to GCS using the notebook's destination."""
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = Path(root) / file
            relative_path = local_file_path.relative_to(local_folder_path)
            destination_parts = [part for part in [gcs_folder_path, str(relative_path)] if part]
            gcs_destination = os.path.join(*destination_parts).replace("\\", "/")

            blob = bucket.blob(gcs_destination)
            blob._chunk_size = 5 * 1024 * 1024
            blob.upload_from_filename(local_file_path)


def main() -> None:
    logger.info("Starting notebook-derived training pipeline")

    df_near = load_near_earth_data()
    X_near_pha, y_pha, X_near_moid, y_moid = prepare_training_datasets(df_near)

    (
        X_train_pha,
        X_test_pha,
        y_train_pha,
        y_test_pha,
        X_train_moid,
        X_test_moid,
        y_train_moid,
        y_test_moid,
    ) = split_training_data(X_near_pha, y_pha, X_near_moid, y_moid)

    pipelines_classification, pipelines_regression = create_model_pipelines()

    if os.getenv("RUN_CROSS_VALIDATION", "0") == "1":
        cross_validate_all(
            X_train_pha,
            y_train_pha,
            X_train_moid,
            y_train_moid,
            pipelines_classification,
            pipelines_regression,
        )

    save_pha_training_stats(X_train_pha)
    model_pha = train_pha_model(
        X_train_pha,
        y_train_pha,
        X_test_pha,
        y_test_pha,
        pipelines_classification,
    )
    model_moid, X_train_moid_clean, y_train_moid_clean = train_moid_model(
        X_train_moid,
        y_train_moid,
        X_test_moid,
        y_test_moid,
        pipelines_regression,
    )

    save_model_artifacts(
        model_pha,
        model_moid,
        X_train_pha,
        y_train_pha,
        X_test_pha,
        y_test_pha,
        X_train_moid_clean,
        y_train_moid_clean,
        X_test_moid,
        y_test_moid,
    )

    if os.getenv("UPLOAD_ARTIFACTS_TO_GCS", "1") == "1":
        logger.info("Uploading artifacts to gs://%s/%s", GCS_BUCKET_NAME, GCS_ARTIFACT_PREFIX)
        upload_local_directory(GCS_BUCKET_NAME, ARTIFACT_PATH, GCS_ARTIFACT_PREFIX)
    else:
        logger.info("Skipping GCS upload because UPLOAD_ARTIFACTS_TO_GCS is not 1")

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
