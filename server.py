import joblib
import pydantic
from fastapi import FastAPI
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level= logging.INFO(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers= [logging.FileHandler('app.log'), logging.StreamHandler()])


app = FastAPI()

['H',
 'diameter_km',
 'size_category',
 'albedo',
 'rot_per_h',
 'class',
 'eccentricity',
 'semimajor_axis_au',
 'inclination_deg',
 'perihelion_distance_au',
 'aphelion_distance_au',
 'orbital_period_days',
 'moid_au',
 'mean_motion_deg_day',
 'condition_code',
 'data_arc']

class neaFeatures(pydantic.BaseModel):
    H: float
    diameter_km: float
    size_category: str
    albedo: float
    rot_per_h: float
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

with open('./artifacts/best_model.pkl', 'rb') as file:
    model = joblib.load(file)

with open('./artifacts/best_model_columntransformer.pkl', 'rb') as file:
    column_transformer = joblib.load(file)

print(model)


@app.get('/')
def home():
    return {"Welcome to the Near Earth Asteroid Hazard Prediction Service"}

@app.predict('/predict_potential_hazard')
def potential_hazard(features: neaFeatures):
    logging.info(f"Received features {features}")

    try:
        df = pd.DataFrame([features.model_dump()])

        X = column_transformer(df)
        logging.debug(f"X.shape: {X.shape}")

        prediction = model.predict(X)

        return {"prediction": prediction}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"Error making prediction"}




