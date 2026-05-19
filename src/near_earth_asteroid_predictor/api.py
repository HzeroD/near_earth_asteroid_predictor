import os
import time
import json
import joblib
import logging
import pandas as pd
from google.cloud.storage import Client, transfer_manager
from pydantic import BaseModel, Field
from schemas import neaFeatures_Pha, neaFeatures_Moid, neaFeatures_Mag, PhaBatch
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

logging.basicConfig(level= logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers= [logging.FileHandler('app.log'), logging.StreamHandler()])
# storage_client = Client()
# bucket = storage_client.bucket("project-3e6b348d-e2ae-4a47-9af_cloudbuild")
# print(bucket)
#ARTIFACTS_PATH = Path("../artifacts").parent.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", PROJECT_ROOT / "artifacts")).resolve()
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
#ARTIFACTS_PATH.parent.parent.mkdir(parents=True, exist_ok=True)

# print(ARTIFACTS_PATH)
# results = transfer_manager.download_many_to_path(
#     bucket = bucket,
#     blob_names = ["best_model.pkl", "best_model_abs_mag.pkl", "best_model_columntransformer.pkl", "column_transformer_abs_mag.pkl", "column_transformer_moid.pkl", "moid_bbest_model.pkl"],
#     destination_directory= ARTIFACTS_PATH,
#     blob_name_prefix="artifacts/"
# )

# print(results)
# print(Path(__file__).resolve().parents[2])








# Inference logging path for monitoring
INFERENCE_LOG_PATH = Path("./logs/inference_events.jsonl")
INFERENCE_LOG_PATH_MOID = Path("./logs/inference_events_moid.jsonl")


# inference event logging function
def log_inference_event(event: dict) -> None:
    if event["model_name"] == "pha":
        INFERENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with INFERENCE_LOG_PATH.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, default=str) + "\n")
    if event["model_name"] == "moid":
        INFERENCE_LOG_PATH_MOID.parent.mkdir(parents=True, exist_ok=True)
        with INFERENCE_LOG_PATH_MOID.open('a', encoding='utf-8') as f:
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
    # Startup: Load model
    global model_pha, model_moid, model_abs_mag
    global column_transformer_pha, column_transformer_moid, column_transformer_mag
    
    try:
            model_pha = joblib.load(ARTIFACTS_PATH / "best_model.pkl")
            model_moid = joblib.load(ARTIFACTS_PATH / "moid_bbest_model.pkl")
            model_abs_mag = joblib.load( ARTIFACTS_PATH / "best_model_abs_mag.pkl")

            column_transformer_pha = joblib.load(ARTIFACTS_PATH / "best_model_columntransformer.pkl")
            column_transformer_moid = joblib.load(ARTIFACTS_PATH / "column_transformer_moid.pkl")
            column_transformer_mag = joblib.load(ARTIFACTS_PATH / "column_transformer_abs_mag.pkl")
        
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
def potential_hazard(features: neaFeatures_Pha | list[neaFeatures_Pha]):

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
            trans = column_transformer_pha.transform(df).tolist()

            X = pd.DataFrame(trans, columns=[f"{col}" for col in column_transformer_pha.get_feature_names_out()])

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
        df_transformed = column_transformer_moid.transform(df).tolist()

        X = pd.DataFrame(df_transformed, columns=[f"{col}" for col in column_transformer_moid.get_feature_names_out() ])
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
            print(type(f.model_dump()))
            df = pd.DataFrame([f.model_dump()])
            df_transformed = column_transformer_moid.transform(df).tolist()

            X = pd.DataFrame(df_transformed, columns=[f"{col}" for col in column_transformer_moid.get_feature_names_out()])
            prediction = model_moid.predict(X)
            prediction = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
            event["prediction"] = prediction
            
        except Exception as e:
            logging.error(f"Error making prediction {e}")
            event["error"] = type(e).__name__
            event["status"] = "error"
    
        finally:
            event["latency_ms"] = round(time.perf_counter() - start, 2)
            log_inference_event(event)
            logging.info(f"Inference event logged")
        
        return {"Moid Predictions: ": predictions}




@app.post('/predict_magnitude')
def predict_magnitude(features: neaFeatures_Mag):
    #logging.info(f"Received features {features}")
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
        print(f"MODEL_ABS_MAG: {column_transformer_mag.transform(df)}")
        print(f"X: {X}")
        y_pred = model_abs_mag.predict(X)
        #y_pred = y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
        event["prediction"] = y_pred

        return {"magnitude_prediction": round(y_pred[0], 2)}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        event["status"] = "error"
        event["error"] = type(e).__name__

        return {f"error": "Error making prediction"}
    
    finally:
        event["latency_ms"] = round(time.perf_counter() - started_at, 2)
        log_inference_event(event)
        logging.info(f"Logged inference event {event}")
