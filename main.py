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
'{"pha":false,"H":13.82,"diameter_km":4.2,"size_category":"Large","class_code":"AMO",\
    "eccentricity":0.5712,"semimajor_axis_au":2.474,"inclination_deg":9.4,"perihelion_distance_au":1.061,\
        "aphelion_distance_au":3.89,"orbital_period_days":1420.0,"mean_motion_deg_day":0.2533,"condition_code":0.0,\
            "data_arc":39281.0}'

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

'{"pha":false,"diameter_km":5.7,"size_category":"Large","class_code":"AMO",\
    "eccentricity":0.5055,"semimajor_axis_au":2.149,"inclination_deg":23.96,\
        "perihelion_distance_au":1.063,"aphelion_distance_au":3.24,"orbital_period_days":1150.0,"moid_au":0.0717,\
            "mean_motion_deg_day":0.3128,"condition_code":0.0,"data_arc":26251.0,"distance_au":0.0896649063,\
                "v_rel_kmh":52010.0,"is_future":true}'

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

with open('./artifacts/best_model.pkl', 'rb') as best_model,\
     open('./artifacts/moid_best_model.pkl', 'rb') as moid_best_model,\
     open('./artifacts/best_model_abs_mag.pkl', 'rb') as mag_best_model:
    model_pha = joblib.load(best_model)
    model_moid = joblib.load(moid_best_model)
    model_abs_mag = joblib.load(mag_best_model)

with open('./artifacts/best_model_columntransformer.pkl', 'rb') as bm_trans, \
     open('./artifacts/column_transformer_moid.pkl', 'rb') as bm_moid_trans, \
     open('./artifacts/column_transformer_abs_mag.pkl', 'rb') as bm_mag_trans:
    column_transformer_pha = joblib.load(bm_trans)
    column_transformer_moid = joblib.load(bm_moid_trans)
    column_transformer_mag = joblib.load(bm_mag_trans)

print(model_pha)



@app.get('/')
def home():
    return {"Welcome to the Near Earth Asteroid Hazard Prediction Service"}

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

        prediction = model_pha.predict(X)

        return {f"prediction: {prediction}"}
    
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return {f"Error making prediction"}
    

@app.post('/predict_moid')
def predict_moid(features: neaFeatures_Moid):
    logging.info(f"Receaived features {features}")

    X = pd.DataFrame([features.model_dump()])
    X_temp = column_transformer_moid.transform(X).tolist()

    X_transformed = pd.DataFrame(X_temp, columns=[f"{col}" for col in column_transformer_moid.get_feature_names_out() ])
    print(X_transformed)
    y_pred = model_moid.predict(X_transformed)

    return {f"moid prediction in au: {y_pred}"}


@app.post('/predict_magnitude')
def predict_magnitude(features: neaFeatures_Mag):
    logging.info(f"Received features {features}")

    X = pd.DataFrame([features.model_dump()])
    X_temp = column_transformer_mag.transform(X).tolist()
    
    X_transformed = pd.DataFrame(X_temp, columns=[f"{col}" for col in column_transformer_mag.get_feature_names_out()])

    y_pred = model_abs_mag.predict(X_transformed)

    return {f"Prediction for absolute magnitude: {y_pred}"}

