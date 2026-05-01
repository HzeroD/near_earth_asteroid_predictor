import joblib
from pydantic import BaseModel, Field
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
            model_pha = joblib.load("../artifacts/best_model.pkl")
            model_moid = joblib.load("../artifacts/moid_best_model.pkl")
            model_abs_mag = joblib.load("../artifacts/best_model_abs_mag.pkl")

            column_transformer_pha = joblib.load("../artifacts/best_model_columntransformer.pkl")
            column_transformer_moid = joblib.load("../artifacts/column_transformer_moid.pkl")
            column_transformer_mag = joblib.load("../artifacts/column_transformer_abs_mag.pkl")
        
    except FileNotFoundError as e:
        logging.warning(f"Model files not found during startup: {e}")
    yield


# model_pha = joblib.load("../../artifacts/best_model.pkl")
# model_moid = joblib.load("../../artifacts/moid_best_model.pkl")
# model_abs_mag = joblib.load("../../artifacts/best_model_abs_mag.pkl")

# column_transformer_pha = joblib.load("../../artifacts/best_model_columntransformer.pkl")
# column_transformer_moid = joblib.load("../../artifacts/column_transformer_moid.pkl")
# column_transformer_mag = joblib.load("../../artifacts/column_transformer_abs_mag.pkl")

app = FastAPI(lifespan=lifespan)



class neaFeatures_Pha(BaseModel):
    H: float = Field(ge=0.0)
    diameter_km: float = Field(ge=0.0)
    size_category: str
    class_code: str
    eccentricity: float = Field(ge=0.0)
    semimajor_axis_au: float = Field(ge=0.0)
    inclination_deg: float = Field(ge=0.0)
    perihelion_distance_au: float = Field(ge=0.0)
    aphelion_distance_au: float = Field(ge=0.0)
    orbital_period_days: float = Field(ge=0.0)
    moid_au: float = Field(ge=0.0)
    mean_motion_deg_day: float = Field(ge=0.0)
    condition_code: float = Field(ge=0.0)
    data_arc: float = Field(ge=0.0)


class neaFeatures_Moid(BaseModel):
    pha: int
    H: float = Field(ge=0.0)
    diameter_km: float = Field(ge=0.0)
    size_category: str
    class_code: str
    eccentricity: float = Field(ge=0.0)
    semimajor_axis_au: float = Field(ge=0.0)
    inclination_deg: float = Field(ge=0.0)
    perihelion_distance_au: float = Field(ge=0.0)
    aphelion_distance_au: float = Field(ge=0.0)
    orbital_period_days: float = Field(ge=0.0)
    mean_motion_deg_day: float = Field(ge=0.0)
    condition_code: float = Field(ge=0.0)
    data_arc: float = Field(ge=0.0)


class neaFeatures_Mag(BaseModel):
    pha: int
    H: float = Field(ge=0.0)
    diameter_km: float = Field(ge=0.0)
    size_category: str
    class_code: str
    eccentricity: float = Field(ge=0.0)
    semimajor_axis_au: float = Field(ge=0.0)
    inclination_deg: float = Field(ge=0.0)
    perihelion_distance_au: float = Field(ge=0.0)
    aphelion_distance_au: float = Field(ge=0.0)
    orbital_period_days: float = Field(ge=0.0)
    moid_au: float = Field(ge=0.0)
    mean_motion_deg_day: float = Field(ge=0.0)
    condition_code: float = Field(ge=0.0)
    data_arc: float = Field(ge=0.0)
    distance_au: float = Field(ge=0.0)
    v_rel_kmh: float = Field(ge=0.0)
    is_future: int


@app.get('/')
def home():
    return {"message":"Welcome to the Near Earth Asteroid Hazard Prediction Service"}

@app.post('/predict_pha')
def potential_hazard(features: neaFeatures_Pha):
    logging.info(f"Received features {features}")
    try:
        #print(features.model_dump())
        df = pd.DataFrame([features.model_dump()])
        transformed_df = column_transformer_pha.transform(df).tolist()

        X = pd.DataFrame(transformed_df, columns=[f"{col}" for col in column_transformer_pha.get_feature_names_out()])

        #df = pd.DataFrame(X)
        print(f"X: {X}")
        logging.debug(f"X.shape: {X.shape}")
        print(model_pha)
        y_pred = model_pha.predict(X)

        return {"pha_prediction":  y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred}
    
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

        return {"moid_prediction": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred}
    
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

        return {"magnitude_prediction": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"error": "Error making prediction"}
