import time
import json
import joblib
import logging
import pandas as pd
from google.cloud.storage import Client, transfer_manager
from pydantic import BaseModel, Field
from schemas import neaFeatures_Pha, neaFeatures_Moid, neaFeatures_Mag
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

logging.basicConfig(level= logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers= [logging.FileHandler('app.log'), logging.StreamHandler()])
storage_client = Client()
bucket = storage_client.bucket("project-3e6b348d-e2ae-4a47-9af-artifacts")
ARTIFACTS_PATH = Path("./artifacts/")
ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)

results = transfer_manager.download_many_to_path(
    bucket = bucket,
    blob_names = ["best_model.pkl", "best_model_abs_mag.pkl", "best_model_columntransformer.pkl", "column_transformer_abs_mag.pkl", "column_transformer_moid.pkl", "moid_best_model.pkl"],
    destination_directory= ARTIFACTS_PATH,
    blob_name_prefix="artifacts/",
    max_workers=8
)

print(results)








# Inference logging path for monitoring
INFERENCE_LOG_PATH = Path("./logs/inference_events.jsonl")

# inference event logging function
def log_inference_event(event: dict) -> None:
    INFERENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INFERENCE_LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, default=str) + "\n")

# Global model variables
model_pha = None
model_moid = None
model_abs_mag = None
column_transformer_pha = None
column_transformer_moid = None
column_transformer_mag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    global model_pha, model_moid, model_abs_mag
    global column_transformer_pha, column_transformer_moid, column_transformer_mag
    
    try:
            model_pha = joblib.load("./artifacts/best_model.pkl")
            model_moid = joblib.load("../../artifacts/moid_best_model.pkl")
            model_abs_mag = joblib.load("../../artifacts/best_model_abs_mag.pkl")

            column_transformer_pha = joblib.load("./artifacts/best_model_columntransformer.pkl")
            column_transformer_moid = joblib.load("../../artifacts/column_transformer_moid.pkl")
            column_transformer_mag = joblib.load("../../artifacts/column_transformer_abs_mag.pkl")
        
    except FileNotFoundError as e:
        logging.warning(f"Model files not found during startup: {e}")
    yield

# The six commented lines below are to be uncommented when testing locally

# model_pha = joblib.load("../../artifacts/best_model.pkl")
# model_moid = joblib.load("../../artifacts/moid_best_model.pkl")
# model_abs_mag = joblib.load("../../artifacts/best_model_abs_mag.pkl")

# column_transformer_pha = joblib.load("../../artifacts/best_model_columntransformer.pkl")
# column_transformer_moid = joblib.load("../../artifacts/column_transformer_moid.pkl")
# column_transformer_mag = joblib.load("../../artifacts/column_transformer_abs_mag.pkl")

app = FastAPI(lifespan=lifespan)



@app.get('/')
def home():
    return {"message":"Welcome to the Near Earth Asteroid Hazard Prediction Service"}

@app.post('/predict_pha')
def potential_hazard(features: neaFeatures_Pha):

    logging.info(f"Received features {features}")

    started_at = time.perf_counter()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "pha",
        "model_version": "local",
        "features": features.model_dump(),
        "prediction": None,
        "status": "success",
        "latency_ms": None,
        "error": None,
    }

    try:
        df = pd.DataFrame([features.model_dump()])
        transformed_df = column_transformer_pha.transform(df).tolist()

        X = pd.DataFrame(transformed_df, columns=[f"{col}" for col in column_transformer_pha.get_feature_names_out()])
        logging.debug(f"X.shape: {X.shape}")

        y_pred = model_pha.predict(X).tolist() if hasattr(y_pred, 'tolist') else y_pred
        event["prediction"] = y_pred

        return {"pha_prediction":  y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        
        event["status"] = "error"
        event["error"] = type(e).__name__

        return {f"error": "Error making prediction"}

    finally:
        event["latency_ms"] = round(time.perf_counter() - started_at)
        log_inference_event(event)
        logging.info(f"Logged inference event {event}")
    
    

@app.post('/predict_moid')
def predict_moid(features: neaFeatures_Moid):
    logging.info(f"Receaived features {features}")
    started_at = time.perf_counter()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": "moid",
        "model_version": "local",
        "features": features.model_dump(),
        "prediction": None,
        "status": "success",
        "latency_ms": None,
        "error": None,
    }

    try:
        df = pd.DataFrame([features.model_dump()])
        df_transformed = column_transformer_moid.transform(X).tolist()

        X = pd.DataFrame(df_transformed, columns=[f"{col}" for col in column_transformer_moid.get_feature_names_out() ])
        
        y_pred = model_moid.predict(X).tolist() if hasattr(y_pred, 'tolist') else y_pred
        event["prediction"] = y_pred

        return {"moid_prediction": y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        event["status"] = "error"
        event["error"] = type(e).__name__

        return {f"error": "Error making prediction"}
    
    finally:
        event["latency_ms"] = round(time.perf_counter() - started_at, 2)
        log_inference_event(event)
        logging.info(f"Logged inference event {event}")




@app.post('/predict_magnitude')
def predict_magnitude(features: neaFeatures_Mag):
    logging.info(f"Received features {features}")
    started_at = time.perf_counter()
    event = {
        "timezone": datetime.now(timezone.utc).isoformat(),
        "model_name": "magnitude",
        "model_version": "local",
        "features": features.model_dump(),
        "prediction": None,
        "status": "success",
        "latency_ms": None,
        "error": None,
    }
    
    try:

        df = pd.DataFrame([features.model_dump()])
        transformed_df = column_transformer_mag.transform(df).tolist()
        
        X = pd.DataFrame(transformed_df, columns=[f"{col}" for col in column_transformer_mag.get_feature_names_out()])

        y_pred = model_abs_mag.predict(X).tolist() if hasattr(y_pred, 'tolist') else y_pred
        event["prediction"] = y_pred

        return {"magnitude_prediction": y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        event["status"] = "error"
        event["error"] = type(e).__name__

        return {f"error": "Error making prediction"}
    
    finally:
        event["latency_ms"] = round(time.perf_counter() - started_at, 2)
        log_inference_event(event)
        logging.info(f"Logged inference event {event}")
