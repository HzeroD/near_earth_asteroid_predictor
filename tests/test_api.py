"""
Unit tests for Near Earth Asteroid Predictor FastAPI server.

Tests cover:
- Endpoint availability and response codes
- Valid and invalid input handling
- Pydantic model validation
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from near_earth_asteroid_predictor.api import (
    app,
    neaFeatures_Mag,
    neaFeatures_Moid,
    neaFeatures_Pha,
)

# ============================================================================
# TEST CLIENT SETUP
# ============================================================================

client = TestClient(app)

# ============================================================================
# TEST DATA / FIXTURES
# ============================================================================

# Valid PHA features (from README example, slightly modified)
VALID_PHA_FEATURES = {
    "H": 13.82,
    "diameter_km": 4.2,
    "size_category": "Large",
    "class_code": "AMO",
    "eccentricity": 0.5712,
    "semimajor_axis_au": 2.474,
    "inclination_deg": 9.4,
    "perihelion_distance_au": 1.061,
    "aphelion_distance_au": 3.89,
    "orbital_period_days": 1420.0,
    "moid_au": 0.0717,
    "mean_motion_deg_day": 0.2533,
    "condition_code": 0.0,
    "data_arc": 39281.0
}

# Valid MOID features
VALID_MOID_FEATURES = {
    "pha": 0,
    "H": 13.82,
    "diameter_km": 5.7,
    "size_category": "Large",
    "class_code": "AMO",
    "eccentricity": 0.5055,
    "semimajor_axis_au": 2.149,
    "inclination_deg": 23.96,
    "perihelion_distance_au": 1.063,
    "aphelion_distance_au": 3.24,
    "orbital_period_days": 1150.0,
    "mean_motion_deg_day": 0.3128,
    "condition_code": 0.0,
    "data_arc": 26251.0
}

# Valid Magnitude features
VALID_MAGNITUDE_FEATURES = {
    "pha": 0,
    "H": 13.82,
    "diameter_km": 5.7,
    "size_category": "Large",
    "class_code": "AMO",
    "eccentricity": 0.5055,
    "semimajor_axis_au": 2.149,
    "inclination_deg": 23.96,
    "perihelion_distance_au": 1.063,
    "aphelion_distance_au": 3.24,
    "orbital_period_days": 1150.0,
    "moid_au": 0.0717,
    "mean_motion_deg_day": 0.3128,
    "condition_code": 0.0,
    "data_arc": 26251.0,
    "distance_au": 0.0896649063,
    "v_rel_kmh": 52010.0,
    "is_future": 1
}

# ============================================================================
# HOME ENDPOINT TESTS
# ============================================================================

class TestHomeEndpoint:
    """Tests for the home endpoint."""
    
    def test_home_returns_200(self):
        """Test that home endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_home_returns_welcome_message(self):
        """Test that home endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0


# ============================================================================
# PHA PREDICTION ENDPOINT TESTS
# ============================================================================

class TestPhaPredictionEndpoint:
    """Tests for the PHA prediction endpoint."""
    
    def test_predict_pha_valid_input_returns_200(self):
        """Test that valid PHA input returns 200 status code."""
        response = client.post("/predict_pha", json=VALID_PHA_FEATURES)
        assert response.status_code == 200
    
    def test_predict_pha_valid_input_returns_prediction(self):
        """Test that valid PHA input returns prediction in response."""
        response = client.post("/predict_pha", json=VALID_PHA_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_predict_pha_missing_required_field(self):
        """Test that missing required field returns 422 validation error."""
        invalid_features = VALID_PHA_FEATURES.copy()
        del invalid_features["H"]  # Remove required field
        response = client.post("/predict_pha", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_pha_wrong_type_float_field(self):
        """Test that providing string for float field returns 422 validation error."""
        invalid_features = VALID_PHA_FEATURES.copy()
        invalid_features["H"] = "not_a_float"  # Should be float
        response = client.post("/predict_pha", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_pha_wrong_type_string_field(self):
        """Test that providing non-string for string field returns 422 validation error."""
        invalid_features = VALID_PHA_FEATURES.copy()
        invalid_features["size_category"] = 123  # Should be string
        response = client.post("/predict_pha", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_pha_extra_fields_ignored(self):
        """Test that extra fields are ignored by Pydantic model."""
        extra_features = VALID_PHA_FEATURES.copy()
        extra_features["extra_field"] = "should_be_ignored"
        response = client.post("/predict_pha", json=extra_features)
        assert response.status_code == 200
    
    def test_predict_pha_negative_values_accepted(self):
        """Test that negative numeric values are accepted (validation passes)."""
        features_with_negative = VALID_PHA_FEATURES.copy()
        features_with_negative["eccentricity"] = -0.5
        response = client.post("/predict_pha", json=features_with_negative)
        # Response should be 200 (Pydantic accepts it; model may handle or reject)
        assert response.status_code == 200


# ============================================================================
# MOID PREDICTION ENDPOINT TESTS
# ============================================================================

class TestMoidPredictionEndpoint:
    """Tests for the MOID prediction endpoint."""
    
    def test_predict_moid_valid_input_returns_200(self):
        """Test that valid MOID input returns 200 status code."""
        response = client.post("/predict_moid", json=VALID_MOID_FEATURES)
        assert response.status_code == 200
    
    def test_predict_moid_valid_input_returns_prediction(self):
        """Test that valid MOID input returns prediction in response."""
        response = client.post("/predict_moid", json=VALID_MOID_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_predict_moid_missing_required_field(self):
        """Test that missing required field returns 422 validation error."""
        invalid_features = VALID_MOID_FEATURES.copy()
        del invalid_features["pha"]  # Remove required field
        response = client.post("/predict_moid", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_moid_wrong_type_int_field(self):
        """Test that providing non-int for int field returns 422 validation error."""
        invalid_features = VALID_MOID_FEATURES.copy()
        invalid_features["pha"] = "not_an_int"  # Should be int
        response = client.post("/predict_moid", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_moid_multiple_missing_fields(self):
        """Test that multiple missing fields all return 422 validation error."""
        invalid_features = VALID_MOID_FEATURES.copy()
        del invalid_features["pha"]
        del invalid_features["H"]
        response = client.post("/predict_moid", json=invalid_features)
        assert response.status_code == 422


# ============================================================================
# MAGNITUDE PREDICTION ENDPOINT TESTS
# ============================================================================

class TestMagnitudePredictionEndpoint:
    """Tests for the absolute magnitude prediction endpoint."""
    
    def test_predict_magnitude_valid_input_returns_200(self):
        """Test that valid magnitude input returns 200 status code."""
        response = client.post("/predict_magnitude", json=VALID_MAGNITUDE_FEATURES)
        assert response.status_code == 200
    
    def test_predict_magnitude_valid_input_returns_prediction(self):
        """Test that valid magnitude input returns prediction in response."""
        response = client.post("/predict_magnitude", json=VALID_MAGNITUDE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_predict_magnitude_missing_required_field(self):
        """Test that missing required field returns 422 validation error."""
        invalid_features = VALID_MAGNITUDE_FEATURES.copy()
        del invalid_features["distance_au"]  # Remove required field
        response = client.post("/predict_magnitude", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_magnitude_wrong_type_float_field(self):
        """Test that providing non-float for float field returns 422 validation error."""
        invalid_features = VALID_MAGNITUDE_FEATURES.copy()
        invalid_features["distance_au"] = "invalid"  # Should be float
        response = client.post("/predict_magnitude", json=invalid_features)
        assert response.status_code == 422
    
    def test_predict_magnitude_wrong_type_is_future_field(self):
        """Test that providing non-int for is_future returns 422 validation error."""
        invalid_features = VALID_MAGNITUDE_FEATURES.copy()
        invalid_features["is_future"] = "yes"  # Should be int
        response = client.post("/predict_magnitude", json=invalid_features)
        assert response.status_code == 422


# ============================================================================
# CROSS-ENDPOINT CONSISTENCY TESTS
# ============================================================================

class TestEndpointConsistency:
    """Tests for consistency across endpoints."""
    
    def test_only_post_allowed_for_predict_endpoints(self):
        """Test that GET requests to predict endpoints return 405 Method Not Allowed."""
        response = client.get("/predict_pha")
        assert response.status_code == 405
        
        response = client.get("/predict_moid")
        assert response.status_code == 405
        
        response = client.get("/predict_magnitude")
        assert response.status_code == 405
    
    def test_only_get_allowed_for_home_endpoint(self):
        """Test that POST requests to home endpoint return 405 Method Not Allowed."""
        response = client.post("/")
        assert response.status_code == 405
    
    def test_nonexistent_endpoint_returns_404(self):
        """Test that nonexistent endpoint returns 404 Not Found."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


# ============================================================================
# PYDANTIC MODEL VALIDATION TESTS
# ============================================================================

class TestPydanticModels:
    """Tests for Pydantic model validation."""
    
    def test_pha_model_valid_creation(self):
        """Test that valid data creates a neaFeatures_Pha model."""
        model = neaFeatures_Pha(**VALID_PHA_FEATURES)
        assert model.H == 13.82
        assert model.size_category == "Large"
    
    def test_pha_model_dict_conversion(self):
        """Test that neaFeatures_Pha model can be converted to dict."""
        model = neaFeatures_Pha(**VALID_PHA_FEATURES)
        data_dict = model.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["H"] == 13.82
    
    def test_moid_model_valid_creation(self):
        """Test that valid data creates a neaFeatures_Moid model."""
        model = neaFeatures_Moid(**VALID_MOID_FEATURES)
        assert model.pha == 0
        assert model.H == 13.82
    
    def test_magnitude_model_valid_creation(self):
        """Test that valid data creates a neaFeatures_Mag model."""
        model = neaFeatures_Mag(**VALID_MAGNITUDE_FEATURES)
        assert model.is_future == 1
        assert model.distance_au == 0.0896649063


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete API."""
    
    def test_health_check_endpoint_accessible(self):
        """Test that the home endpoint (health check) is accessible."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_all_three_endpoints_functional(self):
        """Test that all three prediction endpoints are functional with valid input."""
        pha_response = client.post("/predict_pha", json=VALID_PHA_FEATURES)
        assert pha_response.status_code == 200
        
        moid_response = client.post("/predict_moid", json=VALID_MOID_FEATURES)
        assert moid_response.status_code == 200
        
        mag_response = client.post("/predict_magnitude", json=VALID_MAGNITUDE_FEATURES)
        assert mag_response.status_code == 200
    
    def test_predictions_have_reasonable_structure(self):
        """Test that predictions return data in expected structure."""
        pha_response = client.post("/predict_pha", json=VALID_PHA_FEATURES)
        pha_data = pha_response.json()
        # Response is a dict with string keys containing predictions
        assert isinstance(pha_data, dict)
        
        moid_response = client.post("/predict_moid", json=VALID_MOID_FEATURES)
        moid_data = moid_response.json()
        assert isinstance(moid_data, dict)
        
        mag_response = client.post("/predict_magnitude", json=VALID_MAGNITUDE_FEATURES)
        mag_data = mag_response.json()
        assert isinstance(mag_data, dict)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_predict_pha_with_zero_values(self):
        """Test that zero values in numeric fields are handled."""
        features_with_zeros = VALID_PHA_FEATURES.copy()
        features_with_zeros["H"] = 0.0
        features_with_zeros["diameter_km"] = 0.0
        response = client.post("/predict_pha", json=features_with_zeros)
        assert response.status_code == 200
    
    def test_predict_pha_with_large_values(self):
        """Test that large numeric values are handled."""
        features_with_large = VALID_PHA_FEATURES.copy()
        features_with_large["orbital_period_days"] = 999999.0
        features_with_large["data_arc"] = 9999999.0
        response = client.post("/predict_pha", json=features_with_large)
        assert response.status_code == 200
    
    def test_predict_pha_with_very_small_float_values(self):
        """Test that very small float values are handled."""
        features_with_small = VALID_PHA_FEATURES.copy()
        features_with_small["eccentricity"] = 0.00001
        response = client.post("/predict_pha", json=features_with_small)
        assert response.status_code == 200
    
    def test_empty_json_body_returns_422(self):
        """Test that empty JSON body returns 422 validation error."""
        response = client.post("/predict_pha", json={})
        assert response.status_code == 422
    
    def test_null_required_field_returns_422(self):
        """Test that null value for required field returns 422 validation error."""
        invalid_features = VALID_PHA_FEATURES.copy()
        invalid_features["H"] = None
        response = client.post("/predict_pha", json=invalid_features)
        assert response.status_code == 422
