import joblib
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import numpy as np
import logging
from fastapi.exceptions import RequestValidationError

logging.basicConfig(level= logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers= [logging.FileHandler('app.log'), logging.StreamHandler()])


app = FastAPI()




{"H",
 "diameter_km",
 "size_category",
 "albedo",
 "rot_per_h",
 "class_code",
 "eccentricity",
 "semimajor_axis_au",
 "inclination_deg",
 "perihelion_distance_au",
 "aphelion_distance_au",
 "orbital_period_days",
 "moid_au",
 "mean_motion_deg_day",
 "condition_code",
 "data_arc"}

class neaFeatures(BaseModel):
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

@app.post('/predict')
def potential_hazard(features: neaFeatures):
    logging.info(f"Received features {features}")
    print(pd.DataFrame([features.dict()]).loc[0])
    try:
        print(features.model_dump())
        df = pd.DataFrame([features.model_dump()])
        transformed_df = column_transformer.transform(df).tolist()

        X = pd.DataFrame(transformed_df, columns=[f"{col}" for col in column_transformer.get_feature_names_out()])

        #df = pd.DataFrame(X)
        print(f"X: {X}")
        logging.debug(f"X.shape: {X.shape}")

        prediction = model.predict(X)

        return {f"prediction: {prediction}"}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"Error making prediction"}




