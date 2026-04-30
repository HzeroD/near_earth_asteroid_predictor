import joblib
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level= logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers= [logging.FileHandler('app.log'), logging.StreamHandler()])

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
            model_pha = joblib.load(model_pha)
            model_moid = joblib.load(model_moid)
            model_abs_mag = joblib.load(model_abs_mag)

            column_transformer_pha = joblib.load(column_transformer_pha)
            column_transformer_moid = joblib.load(column_transformer_moid)
            column_transformer_mag = joblib.load(column_transformer_mag)
        
    except FileNotFoundError as e:
        logging.warning(f"Model files not found during startup: {e}")
    yield


app = FastAPI(lifespan=lifespan)



class neaFeatures_Pha(BaseModel):
    H: float
    diameter_km: float
    size_category: str
    class_code: str
    eccentricity: float
    semimajor_axis_au: float
    inclination_deg: float
    perihelion_distance_au: float
    aphelion_distance_au: float
    orbital_period_days: float
    moid_au: float
    mean_motion_deg_day: float
    condition_code: float
    data_arc: float


class neaFeatures_Moid(BaseModel):
    pha: int
    H: float
    diameter_km: float
    size_category: str
    class_code: str
    eccentricity: float
    semimajor_axis_au: float
    inclination_deg: float
    perihelion_distance_au: float
    aphelion_distance_au: float
    orbital_period_days: float
    mean_motion_deg_day: float
    condition_code: float
    data_arc: float


class neaFeatures_Mag(BaseModel):
    pha: int
    H: float
    diameter_km: float
    size_category: str
    class_code: str
    eccentricity: float
    semimajor_axis_au: float
    inclination_deg: float
    perihelion_distance_au: float
    aphelion_distance_au: float
    orbital_period_days: float
    moid_au: float
    mean_motion_deg_day: float
    condition_code: float
    data_arc: float
    distance_au: float
    v_rel_kmh: float
    is_future: int


@app.get('/')
def home():
    return {"message":"Welcome to the Near Earth Asteroid Hazard Prediction Service"}

@app.post('/predict_pha')
def potential_hazard(features: neaFeatures_Pha):
    logging.info(f"Received features {features}")
    print(pd.DataFrame([features.dict()]).loc[0])
    try:
        print(features.model_dump())
        df = pd.DataFrame([features.model_dump()])
        transformed_df = column_transformer_pha.transform(df).tolist()

        X = pd.DataFrame(transformed_df, columns=[f"{col}" for col in column_transformer_pha.get_feature_names_out()])

        #df = pd.DataFrame(X)
        print(f"X: {X}")
        logging.debug(f"X.shape: {X.shape}")

        y_pred = model_pha.predict(X)

        return {"pha_prediction":  y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"error": "Error making prediction"}

    
    

@app.post('/predict_moid')
def predict_moid(features: neaFeatures_Moid):
    logging.info(f"Receaived features {features}")
    try:
        X = pd.DataFrame([features.model_dump()])
        X_temp = column_transformer_moid.transform(X).tolist()

        X_transformed = pd.DataFrame(X_temp, columns=[f"{col}" for col in column_transformer_moid.get_feature_names_out() ])
        print(X_transformed)
        y_pred = model_moid.predict(X_transformed)

        return {"moid_prediction": y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"error": "Error making prediction"}


@app.post('/predict_magnitude')
def predict_magnitude(features: neaFeatures_Mag):
    logging.info(f"Received features {features}")
    
    try:

        X = pd.DataFrame([features.model_dump()])
        X_temp = column_transformer_mag.transform(X).tolist()
        
        X_transformed = pd.DataFrame(X_temp, columns=[f"{col}" for col in column_transformer_mag.get_feature_names_out()])

        y_pred = model_abs_mag.predict(X_transformed)

        return {"magnitude_prediction": y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"error": "Error making prediction"}
