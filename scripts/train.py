"""
Near Earth Asteroid Predictor - Model Training Pipeline

This script trains three separate models to predict:
1. PHA (Potentially Hazardous Asteroid) - Classification
2. MOID (Minimum Orbit Intersection Distance) - Regression
3. Absolute Magnitude - Regression

Models are evaluated using cross-validation, hyperparameter tuned with GridSearchCV,
and the best models are serialized for serving.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, r2_score
import os
from google.cloud import storage

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = {
    "near_earth": "data/near_earth_asteroids_2025.csv",
    "close_approaches": "data/asteroid_close_approaches_2015_2035.csv",
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
    """Load and preprocess both datasets."""
    logger.info("Loading data...")
    
    # Load datasets
    df_near = pd.read_csv(DATA_PATH["near_earth"])
    df_close = pd.read_csv(DATA_PATH["close_approaches"])
    
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
    df_close['close_approach_date'] = pd.to_datetime(
        df_close['close_approach_date'], 
        format="ISO8601"
    )
    df_near['first_obs'] = pd.to_datetime(df_near['first_obs'], format="ISO8601")
    df_near['last_obs'] = pd.to_datetime(df_near['last_obs'], format="ISO8601")
    
    logger.info(f"Loaded {len(df_near)} NEA records and {len(df_close)} close approaches")
    
    return df_near, df_close


def extract_full_name(full_name_str):
    """Extract asteroid designation from full name string."""
    if '(' in full_name_str:
        return full_name_str[full_name_str.index('('):]
    return np.nan


def merge_and_prepare_datasets(df_near, df_close):
    """Prepare three separate datasets for model training.
    
    Returns:
        X_near, y_pha: Features and target for PHA classification
        X_near_no_moid, y_moid: Features and target for MOID regression
        X_merged, y_magnitude: Features and target for absolute magnitude regression
    """
    logger.info("Preparing datasets...")
    
    # Extract full names and clean
    df_near['full_name'] = df_near['full_name'].apply(extract_full_name)
    df_close['full_name'] = df_close['full_name'].apply(extract_full_name)
    df_close = df_close.dropna(how='any', subset=['full_name'], axis=0)
    df_close.reset_index(drop=True, inplace=True)
    
    # Strip whitespace
    df_near['full_name'] = df_near['full_name'].str.strip()
    df_close['full_name'] = df_close['full_name'].str.strip()
    
    # Add is_future flag
    df_close['is_future'] = np.where(
        df_close['close_approach_date'] >= datetime.now(),
        True,
        False
    )
    
    # ---- Dataset 1: PHA (Classification) ----
    X_near = df_near.drop([
        'spkid', 'full_name', 'pdes', 'name', 'diameter_m', 'moid_lunar_distances',
        'albedo', 'rot_per_h', 'moid_km', 'per_y', 'data_arc_years',
        'diameter_is_estimated', 'pha', 'first_obs', 'last_obs'
    ], axis=1)
    X_near['size_category'] = X_near['size_category'].apply(
        lambda x: x[0:x.index(" ")].strip()
    )
    y_pha = df_near['pha']
    
    # ---- Dataset 2: MOID (Regression) ----
    X_near_no_moid = df_near.drop([
        'spkid', 'full_name', 'pdes', 'name', 'diameter_m', 'moid_lunar_distances',
        'albedo', 'rot_per_h', 'moid_km', 'moid_au', 'per_y', 'data_arc_years',
        'diameter_is_estimated', 'first_obs', 'last_obs'
    ], axis=1)
    X_near_no_moid['size_category'] = X_near_no_moid['size_category'].apply(
        lambda x: x[0:x.index(" ")].strip()
    )
    y_moid = df_near['moid_au']
    
    # ---- Dataset 3: Absolute Magnitude (Regression) ----
    X_near_to_merge = df_near.drop([
        'spkid', 'pdes', 'name', 'diameter_m', 'moid_lunar_distances', 'albedo',
        'rot_per_h', 'moid_km', 'per_y', 'data_arc_years', 'diameter_is_estimated',
        'first_obs', 'last_obs', 'H'
    ], axis=1)
    X_near_to_merge['size_category'] = X_near_to_merge['size_category'].apply(
        lambda x: x[0:x.index(" ")].strip()
    )
    
    X_merged = pd.merge(X_near_to_merge, df_close, on='full_name', how='inner').drop(
        ['dist_km', 'dist_lunar'], 
        axis=1
    )
    y_magnitude = X_merged['absolute_magnitude']
    X_merged.drop([
        "full_name", "designation", "close_approach_date", "velocity_km_s",
        "velocity_infinity_km_s", "distance_min_au", "distance_max_au",
        "absolute_magnitude"
    ], axis=1, inplace=True)
    
    logger.info(f"Dataset 1 (PHA): {X_near.shape}")
    logger.info(f"Dataset 2 (MOID): {X_near_no_moid.shape}")
    logger.info(f"Dataset 3 (Magnitude): {X_merged.shape}")
    
    return X_near, y_pha, X_near_no_moid, y_moid, X_merged, y_magnitude


def train_test_split_all(X_near, y_pha, X_near_no_moid, y_moid, X_merged, y_magnitude):
    """Split all three datasets into train/test sets."""
    logger.info("Performing train/test split...")
    
    X_train_pha, X_test_pha, y_train_pha, y_test_pha = train_test_split(
        X_near, y_pha, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    X_train_moid, X_test_moid, y_train_moid, y_test_moid = train_test_split(
        X_near_no_moid, y_moid,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    X_train_mag, X_test_mag, y_train_mag, y_test_mag = train_test_split(
        X_merged, y_magnitude,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    return (
        X_train_pha, X_test_pha, y_train_pha, y_test_pha,
        X_train_moid, X_test_moid, y_train_moid, y_test_moid,
        X_train_mag, X_test_mag, y_train_mag, y_test_mag
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
    X_train_mag, y_train_mag,
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
    
    # ---- Absolute Magnitude Regression (5-Fold CV) ----
    logger.info("\nAbsolute Magnitude Regression (R2 Score):")
    cv_results['magnitude'] = {}
    for name, pipeline in pipelines_regression.items():
        y_clean = y_train_mag.copy()
        X_clean = X_train_mag.copy()
        
        nan_idx = y_clean[y_clean.isna()].index.tolist()
        if nan_idx:
            y_clean = y_clean.drop(nan_idx, axis=0)
            X_clean = X_clean.drop(nan_idx, axis=0)
        
        scores = cross_val_score(
            pipeline, X_clean, y_clean,
            cv=CV_FOLDS,
            scoring='r2'
        )
        cv_results['magnitude'][name] = scores
        logger.info(f"  {name:30s}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    logger.info("="*70 + "\n")
    return cv_results


# ============================================================================
# MODEL TRAINING - PHA (Classification)
# ============================================================================

def train_pha_model(X_train_pha, y_train_pha, X_test_pha, y_test_pha):
    """Train PHA model with hyperparameter tuning."""
    logger.info("Training PHA model with GridSearchCV...")
    
    preprocessor = create_preprocessor()
    pipeline = make_pipeline(preprocessor, RandomForestClassifier())
    
    param_grid = {
        "randomforestclassifier__n_estimators": [10, 70, 100, 500],
        "randomforestclassifier__max_depth": [2, 5, 7, 10, None],
        "randomforestclassifier__min_samples_leaf": [2, 3, 4]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_FOLDS,
        scoring='f1_micro',
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_pha, y_train_pha)
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_pha)
    test_f1 = f1_score(y_test_pha, y_pred)
    
    logger.info(f"Best PHA parameters: {grid_search.best_params_}")
    logger.info(f"Best CV F1 score: {grid_search.best_score_:.3f}")
    logger.info(f"Test set F1 score: {test_f1:.3f}")
    
    return best_model


# ============================================================================
# MODEL TRAINING - MOID (Regression)
# ============================================================================

def train_moid_model(X_train_moid, y_train_moid, X_test_moid, y_test_moid):
    """Train MOID model with hyperparameter tuning."""
    logger.info("Training MOID model with GridSearchCV...")
    
    # Clean NaN values
    nan_idx = y_train_moid[y_train_moid.isna()].index.tolist()
    X_train_clean = X_train_moid.drop(nan_idx, axis=0).copy()
    y_train_clean = y_train_moid.drop(nan_idx, axis=0).copy()
    
    nan_idx_test = y_test_moid[y_test_moid.isna()].index.tolist()
    X_test_clean = X_test_moid.drop(nan_idx_test, axis=0).copy()
    y_test_clean = y_test_moid.drop(nan_idx_test, axis=0).copy()
    
    preprocessor = create_preprocessor()
    pipeline = make_pipeline(preprocessor, GradientBoostingRegressor())
    
    param_grid = {
        "gradientboostingregressor__n_estimators": [10, 30, 100, 300],
        "gradientboostingregressor__min_samples_leaf": [1, 2, 3],
        "gradientboostingregressor__max_depth": [2, 3, 5, 9]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV_FOLDS,
        scoring='r2',
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_clean, y_train_clean)
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_clean)
    test_r2 = r2_score(y_test_clean, y_pred)
    
    logger.info(f"Best MOID parameters: {grid_search.best_params_}")
    logger.info(f"Best CV R2 score: {grid_search.best_score_:.3f}")
    logger.info(f"Test set R2 score: {test_r2:.3f}")
    
    return best_model


# ============================================================================
# MODEL TRAINING - Absolute Magnitude (Regression)
# ============================================================================

def train_magnitude_model(X_train_mag, y_train_mag, X_test_mag, y_test_mag):
    """Train absolute magnitude model (using RF from cross-validation)."""
    logger.info("Training absolute magnitude model...")
    
    preprocessor = create_preprocessor()
    pipeline = make_pipeline(preprocessor, RandomForestRegressor())
    
    pipeline.fit(X_train_mag, y_train_mag)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test_mag)
    test_r2 = r2_score(y_test_mag, y_pred)
    
    logger.info(f"Test set R2 score: {test_r2:.3f}")
    
    return pipeline


# ============================================================================
# MODEL SERIALIZATION
# ============================================================================

def save_model_artifacts(model_pha, model_moid, model_magnitude):
    """Save trained models and column transformers to disk."""
    logger.info("Saving model artifacts...")
    
    # PHA model and transformer
    joblib.dump(
        model_pha['randomforestclassifier'],
        f'{ARTIFACT_PATH}/best_model.pkl'
    )
    joblib.dump(
        model_pha['columntransformer'],
        f'{ARTIFACT_PATH}/best_model_columntransformer.pkl'
    )
    logger.info("Saved PHA model and transformer")
    
    # MOID model and transformer
    joblib.dump(
        model_moid['gradientboostingregressor'],
        f'{ARTIFACT_PATH}/moid_best_model.pkl'
    )
    joblib.dump(
        model_moid['columntransformer'],
        f'{ARTIFACT_PATH}/column_transformer_moid.pkl'
    )
    logger.info("Saved MOID model and transformer")
    
    # Magnitude model and transformer
    joblib.dump(
        model_magnitude['randomforestregressor'],
        f'{ARTIFACT_PATH}/best_model_abs_mag.pkl'
    )
    joblib.dump(
        model_magnitude['columntransformer'],
        f'{ARTIFACT_PATH}/column_transformer_abs_mag.pkl'
    )
    logger.info("Saved absolute magnitude model and transformer")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Data loading and preprocessing
    df_near, df_close = load_and_preprocess_data()
    X_near, y_pha, X_near_no_moid, y_moid, X_merged, y_magnitude = \
        merge_and_prepare_datasets(df_near, df_close)
    
    # Train-test split
    X_train_pha, X_test_pha, y_train_pha, y_test_pha, \
    X_train_moid, X_test_moid, y_train_moid, y_test_moid, \
    X_train_mag, X_test_mag, y_train_mag, y_test_mag = \
        train_test_split_all(X_near, y_pha, X_near_no_moid, y_moid, X_merged, y_magnitude)
    
    # Create pipelines
    pipelines_classification, pipelines_regression = create_model_pipelines()
    
    # Cross-validation
    cv_results = cross_validate_models(
        X_train_pha, y_train_pha,
        X_train_moid, y_train_moid,
        X_train_mag, y_train_mag,
        pipelines_classification,
        pipelines_regression
    )
    
    # Train models
    logger.info("\n" + "="*70)
    logger.info("MODEL TRAINING AND EVALUATION")
    logger.info("="*70 + "\n")
    
    model_pha = train_pha_model(X_train_pha, y_train_pha, X_test_pha, y_test_pha)
    model_moid = train_moid_model(X_train_moid, y_train_moid, X_test_moid, y_test_moid)
    model_magnitude = train_magnitude_model(X_train_mag, y_train_mag, X_test_mag, y_test_mag)
    
    # Save artifacts
    save_model_artifacts(model_pha, model_moid, model_magnitude)

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



    upload_local_directory('project-3e6b348d-e2ae-4a47-9af_cloudbuild', '../artifacts','artifacts/')
    
    logger.info("\n" + "="*70)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
