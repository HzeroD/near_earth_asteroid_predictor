import os
import time
import json
import joblib
import logging
import uuid
import pandas as pd
from .schemas import neaFeatures_Pha, neaFeatures_Moid, PhaBatch
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
from google.cloud import storage




logging.basicConfig(level= logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers= [logging.FileHandler('app.log'), logging.StreamHandler()])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", PROJECT_ROOT / "artifacts")).resolve()
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

# Inference logging path for monitoring
LOGS_PATH = Path(os.getenv("INFERENCE_LOG_DIR", "./logs"))
INFERENCE_LOG_PATHS = {
    "pha": LOGS_PATH / "inference_events_pha.jsonl",
    "moid": LOGS_PATH / "inference_events_moid.jsonl",
}


# inference event local logging function
def log_inference_event(event: dict) -> None:
    log_path = INFERENCE_LOG_PATHS.get(event.get("model_name"))
    if log_path is None:
        logging.warning("Skipping inference log for unknown model: %s", event.get("model_name"))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, default=str) + "\n")



INFERENCE_LOG_BUCKET = os.getenv("INFERENCE_LOG_BUCKET")
INFERENCE_LOG_BUCKET_PHA = os.getenv("INFERENCE_LOG_BUCKET_PHA")
INFERENCE_LOG_BUCKET_MOID = os.getenv("INFERENCE_LOG_BUCKET_MOID")
INFERENCE_LOG_PREFIX = os.getenv("INFERENCE_LOG_PREFIX", "inference_logs")

def write_inference_event_to_gcs(event: dict) -> None:
    if not INFERENCE_LOG_BUCKET:
        return
    
    # _, log_name = INFERENCE_LOG_BUCKET.rsplit("/",1)
    model_name = event.get("model_name","unknown")
    blob_name = ""

    if model_name == "pha":
        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())
        date_part = timestamp[:10]
        blob_name = (
            f"date={date_part}/" 
            f"model={event.get("model_name", "unknown")}/"
            f"{event.get("request_id")}.jsonl"
        )
    elif model_name == "moid":
        if not INFERENCE_LOG_BUCKET_MOID:
            return
        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())
        date_part = timestamp[:10]
        blob_name = (
            f"date={date_part}/"
            f"model={event.get("model_name", "unknown")}/"
            f"{model_name}.jsonl"
        )
    
    bucket = storage.Client().bucket(INFERENCE_LOG_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(event, default=str) + "\n",
        content_type="application/x-ndjson"
    )

# inference event local logging function
def log_inference_event(event: dict) -> None:
    log_path = INFERENCE_LOG_PATHS.get(event.get("model_name"))
    if log_path is None:
        logging.warning("Skipping inference log for unknown model: %s", event.get("model_name"))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, default=str) + "\n")
    
    try:
        write_inference_event_to_gcs()
    except Exception:
        logging.exception("Failed to write inference event to GCS")





# Global model variables
model_pha = None
model_moid = None
columntransformer_pha = None
columntransformer_moid = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model_pha, model_moid
    global columntransformer_pha, columntransformer_moid
    
    try:
        model_pha = joblib.load(ARTIFACTS_PATH / "best_model_pha.pkl")
        model_moid = joblib.load(ARTIFACTS_PATH / "best_model_moid.pkl")

        columntransformer_pha = joblib.load(ARTIFACTS_PATH / "columntransformer_pha.pkl")
        columntransformer_moid = joblib.load(ARTIFACTS_PATH / "columntransformer_moid.pkl")

    except FileNotFoundError as e:
        logging.warning(f"Model files not found during startup: {e}")

    yield

# The commented lines below are to be uncommented when testing locally

# model_pha = joblib.load("../../artifacts/best_model.pkl")
# model_moid = joblib.load("../../artifacts/moid_best_model.pkl")

# column_transformer_pha = joblib.load("../../artifacts/best_model_columntransformer.pkl")
# column_transformer_moid = joblib.load("../../artifacts/column_transformer_moid.pkl")

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
        transformed_df = columntransformer_pha.transform(df).tolist()

        X = pd.DataFrame(transformed_df, columns=[f"{col}" for col in columntransformer_pha.get_feature_names_out()])
        logging.debug(f"X.shape: {X.shape}")

        y_pred = model_pha.predict(X)
        y_pred = y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
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
    
@app.post('/predict_pha_batch')
def predict_pha_batch(features: list[neaFeatures_Pha]):\
    
    predictions = []
    for f in features:
        
        start = time.perf_counter()
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": "pha",
            "model_version": "local",
            "features": f.model_dump(),
            "prediction": None,
            "status": "success",
            "latency_ms": None,
            "error": None
        }

        try:
            df = pd.DataFrame([f.model_dump()])
            trans = columntransformer_pha.transform(df).tolist()

            X = pd.DataFrame(trans, columns=[f"{col}" for col in columntransformer_pha.get_feature_names_out()])

            y_pred = model_pha.predict(X)
            y_pred = y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred

            event['prediction'] = y_pred
            predictions.append(y_pred)

        except Exception as e:
            event['status'] = 'error'
            event['error'] = type(e).__name__

            return {"error": "Error making prediction"}
        
        finally:
            event['latency_ms'] = round((time.perf_counter() - start), 2)
            log_inference_event(event)
            logging.info(f"Logged inference event {event}")
        
    return {"pha predictions": predictions}

    

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
        df_transformed = columntransformer_moid.transform(df).tolist()

        X = pd.DataFrame(df_transformed, columns=[f"{col}" for col in columntransformer_moid.get_feature_names_out() ])
        logging.debug(f"X.shape: {X.shape}")

        prediction = model_moid.predict(X)
        prediction = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        event["prediction"] = prediction

        return {"moid_prediction": prediction}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        event["status"] = "error"
        event["error"] = type(e).__name__

        return {f"error": "Error making prediction"}
    
    finally:
        event["latency_ms"] = round(time.perf_counter() - started_at, 2)
        log_inference_event(event)
        logging.info(f"Logged inference event {event}")


@app.post('/predict_moid_batch')
def predict_moid_batch(features: list[neaFeatures_Moid]):
    logging.info(f"Received features {features}")
    predictions = []

    start = time.perf_counter()
    for f in features:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": "moid",
            "model_version": "local",
            "features": f.model_dump(),
            "prediction": None,
            "status": "success",
            "latency_ms": None,
            "error": None
        }

        try:
            df = pd.DataFrame([f.model_dump()])
            df_transformed = columntransformer_moid.transform(df).tolist()

            X = pd.DataFrame(df_transformed, columns=[f"{col}" for col in columntransformer_moid.get_feature_names_out()])
            prediction = model_moid.predict(X)
            prediction = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
            event["prediction"] = prediction
            predictions.append(prediction)
            
        except Exception as e:
            logging.error(f"Error making prediction {e}")
            event["error"] = type(e).__name__
            event["status"] = "error"
    
        finally:
            event["latency_ms"] = round(time.perf_counter() - start, 2)
            log_inference_event(event)
            logging.info(f"Inference event logged")
        
    return {"moid_predictions": predictions}
