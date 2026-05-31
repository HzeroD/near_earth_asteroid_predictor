"""
Near Earth Asteroid Predictor - Model Training Pipeline

This script trains two separate models to predict:
1. PHA (Potentially Hazardous Asteroid) - Classification
2. MOID (Minimum Orbit Intersection Distance) - Regression

Models are evaluated using cross-validation and serialized for serving.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, r2_score
import os
from google.cloud import storage

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = {
    "near_earth": "data/near_earth_asteroids_2025.csv",
}

ARTIFACT_PATH = "./artifacts"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """Load and preprocess the near-Earth asteroid dataset."""
    logger.info("Loading data...")
    
    df_near = pd.read_csv(DATA_PATH["near_earth"])
    
    # Rename columns in near-Earth asteroids dataset
    rename_mapping = {
        'q': 'perihelion_distance_au',
        'ad': 'aphelion_distance_au',
        'e': 'eccentricity',
        'a': 'semimajor_axis_au',
        'i': 'inclination_deg',
        'per': 'orbital_period_days',
        'n': 'mean_motion_deg_day',
        'rot_per': 'rot_per_h',
        'class': 'class_code'
    }
    df_near.rename(rename_mapping, axis=1, inplace=True)
    
    # Convert date columns
    df_near['first_obs'] = pd.to_datetime(df_near['first_obs'], format="ISO8601")
    df_near['last_obs'] = pd.to_datetime(df_near['last_obs'], format="ISO8601")
    
    logger.info(f"Loaded {len(df_near)} NEA records")
    
    return df_near


def prepare_training_datasets(df_near):
    """Prepare PHA and MOID datasets for model training.
    
    Returns:
        X_near_pha, y_pha: Features and target for PHA classification
        X_near_moid, y_moid: Features and target for MOID regression
    """
    logger.info("Preparing datasets...")
    
    # ---- PHA (Classification) ----
    X_near_pha = df_near.drop([
        'spkid', 'full_name', 'pdes', 'name', 'diameter_m', 'moid_lunar_distances',
        'albedo', 'rot_per_h', 'moid_km', 'per_y', 'data_arc_years',
        'diameter_is_estimated', 'pha', 'first_obs', 'last_obs'
    ], axis=1)
    X_near_pha['size_category'] = X_near_pha['size_category'].apply(
        lambda x: x[0:x.index(" ")].strip()
    )
    y_pha = df_near['pha']
    
    # ---- MOID (Regression) ----
    X_near_moid = df_near.drop([
        'spkid', 'full_name', 'pdes', 'name', 'diameter_m', 'moid_lunar_distances',
        'albedo', 'rot_per_h', 'moid_km', 'moid_au', 'per_y', 'data_arc_years',
        'diameter_is_estimated', 'first_obs', 'last_obs'
    ], axis=1)
    X_near_moid['size_category'] = X_near_moid['size_category'].apply(
        lambda x: x[0:x.index(" ")].strip()
    )
    y_moid = df_near['moid_au']
    
    logger.info(f"Dataset 1 (PHA): {X_near_pha.shape}")
    logger.info(f"Dataset 2 (MOID): {X_near_moid.shape}")
    
    return X_near_pha, y_pha, X_near_moid, y_moid


def train_test_split_all(X_near_pha, y_pha, X_near_moid, y_moid):
    """Split PHA and MOID datasets into train/test sets."""
    logger.info("Performing train/test split...")
    
    X_train_pha, X_test_pha, y_train_pha, y_test_pha = train_test_split(
        X_near_pha, y_pha, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    X_train_moid, X_test_moid, y_train_moid, y_test_moid = train_test_split(
        X_near_moid, y_moid,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    return (
        X_train_pha, X_test_pha, y_train_pha, y_test_pha,
        X_train_moid, X_test_moid, y_train_moid, y_test_moid
    )


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def create_preprocessor():
    """Create a ColumnTransformer for preprocessing.
    
    Applies OneHotEncoding to categorical features and StandardScaling to numeric.
    """
    categorical_transformer = Pipeline(steps=[
        ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('simple_imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('standard_scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical_transformer', categorical_transformer,
             make_column_selector(dtype_include=['object', 'category', 'str'])),
            ('numeric_transformer', numeric_transformer,
             make_column_selector(dtype_include='number'))
        ]
    )
    
    return preprocessor


# ============================================================================
# MODEL PIPELINES
# ============================================================================

def create_model_pipelines():
    """Create classification and regression model pipelines."""
    preprocessor = create_preprocessor()
    
    # Classification pipelines
    pipelines_classification = {
        "Random Forest Classifier": make_pipeline(
            preprocessor, 
            RandomForestClassifier()
        ),
        "Logistic Regression": make_pipeline(
            preprocessor,
            LogisticRegression()
        ),
        "Gradient Boosting Classifier": make_pipeline(
            preprocessor,
            GradientBoostingClassifier()
        )
    }
    
    # Regression pipelines
    pipelines_regression = {
        "Random Forest Regressor": make_pipeline(
            preprocessor,
            RandomForestRegressor()
        ),
        "Decision Tree Regressor": make_pipeline(
            preprocessor,
            DecisionTreeRegressor()
        ),
        "Gradient Boosting Regressor": make_pipeline(
            preprocessor,
            GradientBoostingRegressor()
        ),
        "Linear Regression": make_pipeline(
            preprocessor,
            LinearRegression()
        )
    }
    
    return pipelines_classification, pipelines_regression


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate_models(
    X_train_pha, y_train_pha,
    X_train_moid, y_train_moid,
    pipelines_classification,
    pipelines_regression
):
    """Cross-validate all models and report scores."""
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*70)
    
    stratified_kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
    cv_results = {}
    
    # ---- PHA Classification (Stratified KFold) ----
    logger.info("\nPHA Classification (F1 Score):")
    cv_results['pha'] = {}
    for name, pipeline in pipelines_classification.items():
        y_clean = y_train_pha.copy()
        X_clean = X_train_pha.copy()
        
        # Remove NaN values
        nan_idx = y_clean[y_clean.isna()].index.tolist()
        if nan_idx:
            y_clean = y_clean.drop(nan_idx, axis=0)
            X_clean = X_clean.drop(nan_idx, axis=0)
        
        scores = cross_val_score(
            pipeline, X_clean, y_clean,
            cv=stratified_kf,
            scoring='f1_micro'
        )
        cv_results['pha'][name] = scores
        logger.info(f"  {name:30s}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # ---- MOID Regression (5-Fold CV) ----
    logger.info("\nMOID Regression (R2 Score):")
    cv_results['moid'] = {}
    for name, pipeline in pipelines_regression.items():
        y_clean = y_train_moid.copy()
        X_clean = X_train_moid.copy()
        
        nan_idx = y_clean[y_clean.isna()].index.tolist()
        if nan_idx:
            y_clean = y_clean.drop(nan_idx, axis=0)
            X_clean = X_clean.drop(nan_idx, axis=0)
        
        scores = cross_val_score(
            pipeline, X_clean, y_clean,
            cv=CV_FOLDS,
            scoring='r2'
        )
        cv_results['moid'][name] = scores
        logger.info(f"  {name:30s}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    logger.info("="*70 + "\n")
    return cv_results


# ============================================================================
# MODEL TRAINING - PHA (Classification)
# ============================================================================

def train_pha_model(X_train_pha, y_train_pha, X_test_pha, y_test_pha):
    """Train PHA model using the notebook-selected Random Forest Classifier."""
    logger.info("Training PHA model...")
    
    preprocessor = create_preprocessor()
    best_model = make_pipeline(preprocessor, RandomForestClassifier())
    best_model.fit(X_train_pha, y_train_pha)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_pha)
    test_f1 = f1_score(y_test_pha, y_pred)
    
    logger.info(f"Test set F1 score: {test_f1:.3f}")
    
    return best_model


# ============================================================================
# MODEL TRAINING - MOID (Regression)
# ============================================================================

def train_moid_model(X_train_moid, y_train_moid, X_test_moid, y_test_moid):
    """Train MOID model using the notebook-selected Random Forest Regressor."""
    logger.info("Training MOID model...")
    
    # Clean NaN values
    nan_idx = y_train_moid[y_train_moid.isna()].index.tolist()
    X_train_clean = X_train_moid.drop(nan_idx, axis=0).copy()
    y_train_clean = y_train_moid.drop(nan_idx, axis=0).copy()
    
    nan_idx_test = y_test_moid[y_test_moid.isna()].index.tolist()
    X_test_clean = X_test_moid.drop(nan_idx_test, axis=0).copy()
    y_test_clean = y_test_moid.drop(nan_idx_test, axis=0).copy()
    
    preprocessor = create_preprocessor()
    best_model = make_pipeline(preprocessor, RandomForestRegressor())
    best_model.fit(X_train_clean, y_train_clean)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_clean)
    test_r2 = r2_score(y_test_clean, y_pred)
    
    logger.info(f"Test set R2 score: {test_r2:.3f}")
    
    return best_model


# ============================================================================
# MODEL SERIALIZATION
# ============================================================================

def save_model_artifacts(
    model_pha,
    model_moid,
    X_train_pha,
    y_train_pha,
    X_test_pha,
    y_test_pha,
    X_train_moid,
    y_train_moid,
    X_test_moid,
    y_test_moid,
):
    """Save trained models, column transformers, and monitoring reference data."""
    logger.info("Saving model artifacts...")
    os.makedirs(ARTIFACT_PATH, exist_ok=True)
    
    # PHA model and transformer
    joblib.dump(
        model_pha['randomforestclassifier'],
        f'{ARTIFACT_PATH}/best_model_pha.pkl'
    )
    joblib.dump(
        model_pha['columntransformer'],
        f'{ARTIFACT_PATH}/columntransformer_pha.pkl'
    )
    logger.info("Saved PHA model and transformer")
    
    # MOID model and transformer
    joblib.dump(
        model_moid['randomforestregressor'],
        f'{ARTIFACT_PATH}/best_model_moid.pkl'
    )
    joblib.dump(
        model_moid['columntransformer'],
        f'{ARTIFACT_PATH}/columntransformer_moid.pkl'
    )
    logger.info("Saved MOID model and transformer")

    # Monitoring reference and test sets
    joblib.dump(X_train_pha, f'{ARTIFACT_PATH}/X_train_pha.pkl')
    joblib.dump(y_train_pha, f'{ARTIFACT_PATH}/y_train_pha.pkl')
    joblib.dump(X_test_pha, f'{ARTIFACT_PATH}/X_test_pha.pkl')
    joblib.dump(y_test_pha, f'{ARTIFACT_PATH}/y_test_pha.pkl')
    joblib.dump(X_train_moid, f'{ARTIFACT_PATH}/X_train_moid.pkl')
    joblib.dump(y_train_moid, f'{ARTIFACT_PATH}/y_train_moid.pkl')
    joblib.dump(X_test_moid, f'{ARTIFACT_PATH}/X_test_moid.pkl')
    joblib.dump(y_test_moid, f'{ARTIFACT_PATH}/y_test_moid.pkl')
    joblib.dump(X_train_moid.describe(), f'{ARTIFACT_PATH}/X_train_moid_stats.pkl')
    logger.info("Saved monitoring reference and test artifacts")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Data loading and preprocessing
    df_near = load_and_preprocess_data()
    X_near_pha, y_pha, X_near_moid, y_moid = prepare_training_datasets(df_near)
    
    # Train-test split
    X_train_pha, X_test_pha, y_train_pha, y_test_pha, \
    X_train_moid, X_test_moid, y_train_moid, y_test_moid = \
        train_test_split_all(X_near_pha, y_pha, X_near_moid, y_moid)
    
    # Create pipelines
    pipelines_classification, pipelines_regression = create_model_pipelines()
    
    # Cross-validation
    cross_validate_models(
        X_train_pha, y_train_pha,
        X_train_moid, y_train_moid,
        pipelines_classification,
        pipelines_regression
    )
    
    # Train models
    logger.info("\n" + "="*70)
    logger.info("MODEL TRAINING AND EVALUATION")
    logger.info("="*70 + "\n")
    
    model_pha = train_pha_model(X_train_pha, y_train_pha, X_test_pha, y_test_pha)
    model_moid = train_moid_model(X_train_moid, y_train_moid, X_test_moid, y_test_moid)
    
    # Save artifacts
    save_model_artifacts(
        model_pha,
        model_moid,
        X_train_pha,
        y_train_pha,
        X_test_pha,
        y_test_pha,
        X_train_moid,
        y_train_moid,
        X_test_moid,
        y_test_moid,
    )

    def upload_local_directory(bucket_name, local_folder_path, gcs_folder_path=None):
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                print(f"local_file_path: {local_file_path}")

                relative_path = os.path.relpath(local_file_path, local_folder_path)
                print(f"relative_path: {relative_path}")
                gcs_destination = os.path.join(gcs_folder_path, relative_path ).replace("\\", "/")
                print(f"gcs_destination: {gcs_destination}")

                blob = bucket.blob(gcs_destination)
                blob.upload_from_filename(local_file_path)
                print(f"Uploaded {file} to {gcs_destination}")



    upload_local_directory('project-3e6b348d-e2ae-4a47-9af_cloudbuild', ARTIFACT_PATH, 'artifacts/')
    
    logger.info("\n" + "="*70)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
