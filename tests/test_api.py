import pytest
from fastapi.testclient import TestClient
from project_pwc.api.app import app

client = TestClient(app)

def test_predict_salary()-> None:
    payload = {
        "years_experience": 5.0,
        "gender_female": 1,
        "gender_male": 0,
        "education_level_ordinal": 2,
        "experience_level_ordinal": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_salary" in data
    assert isinstance(data["predicted_salary"], float)
