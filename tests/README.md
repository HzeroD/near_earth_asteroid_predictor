# API Tests

This directory contains unit and integration tests for the FastAPI server.

## Test Coverage

### Endpoint Tests

- **Home Endpoint** (`TestHomeEndpoint`): Tests the welcome endpoint
- **PHA Prediction** (`TestPhaPredictionEndpoint`): Tests the potentially hazardous asteroid classifier
- **MOID Prediction** (`TestMoidPredictionEndpoint`): Tests the minimum orbit intersection distance regressor
- **Magnitude Prediction** (`TestMagnitudePredictionEndpoint`): Tests the absolute magnitude regressor

### Validation Tests

- **Pydantic Models** (`TestPydanticModels`): Tests input model validation
- **Input Validation**: Tests for missing fields, wrong types, null values
- **HTTP Method Validation**: Tests that endpoints only accept correct HTTP methods (GET/POST)

### Integration Tests

- **Endpoint Consistency** (`TestEndpointConsistency`): Tests behavior across endpoints
- **Integration** (`TestIntegration`): Tests that all endpoints work together
- **Edge Cases** (`TestEdgeCases`): Tests boundary conditions (zero values, large values, small floats)

## Running Tests

### Install test dependencies

```bash
pip install pytest
```

### Run all tests

```bash
pytest
```

### Run tests with verbose output

```bash
pytest -v
```

### Run specific test file

```bash
pytest tests/test_main.py -v
```

### Run specific test class

```bash
pytest tests/test_main.py::TestPhaPredictionEndpoint -v
```

### Run specific test function

```bash
pytest tests/test_main.py::TestPhaPredictionEndpoint::test_predict_pha_valid_input_returns_200 -v
```

### Run tests and show print statements

```bash
pytest -v -s
```

### Run tests with coverage report

```bash
pip install pytest-cov
pytest --cov=. tests/
```

## Test Data

The test fixtures include realistic asteroid data based on the project's examples:

- `VALID_PHA_FEATURES`: Features for PHA (hazard) classification
- `VALID_MOID_FEATURES`: Features for MOID (distance) regression
- `VALID_MAGNITUDE_FEATURES`: Features for absolute magnitude regression

## Test Organization

Tests are organized into logical groups:

1. **Home Endpoint Tests**: Basic functionality
2. **Endpoint-Specific Tests**: Each prediction endpoint tested independently
3. **Cross-Endpoint Tests**: Consistency across endpoints
4. **Pydantic Model Tests**: Input validation at the model level
5. **Integration Tests**: Multiple components working together
6. **Edge Case Tests**: Boundary conditions and unusual inputs

## What Each Test Does

### Status Code Tests

Verify that endpoints return appropriate HTTP status codes:

- `200 OK`: Valid request
- `422 Unprocessable Entity`: Invalid input (validation error)
- `404 Not Found`: Endpoint doesn't exist
- `405 Method Not Allowed`: Wrong HTTP method

### Input Validation Tests

Test that Pydantic models properly validate:

- Required fields present
- Correct data types
- Acceptable value ranges

### Response Structure Tests

Verify that responses have expected structure and content.

## Notes

- Tests require `main.py` and its dependencies to be importable
- The tests use FastAPI's `TestClient` for synchronous testing
- Model artifacts must be available for the server to start (for the actual predictions to work)
- Some tests may fail if artifact files are missing, but validation tests will still pass
