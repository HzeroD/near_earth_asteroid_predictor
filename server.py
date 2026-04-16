import joblib
import pydantic
from fastapi import FastAPI
import pandas as pd
import numpy as np


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





