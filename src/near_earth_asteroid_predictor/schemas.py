
from pydantic import BaseModel, Field

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