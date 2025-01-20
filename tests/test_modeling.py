import os
import pytest
from project_pwc.modeling.train_and_save_best_model import train_and_save_model
from project_pwc.config import PROCESSED_DATA_DIR
import joblib

def test_train_and_save_model():

    model_out = "models/test_rf_salary.joblib"
    train_and_save_model(
        input_csv=(PROCESSED_DATA_DIR / "dataset_features.csv"),
        output_model=model_out
    )

    assert os.path.exists(model_out), f"No se generó {model_out}"

    rf_model = joblib.load(model_out)
    from sklearn.ensemble import RandomForestRegressor
    assert isinstance(rf_model, RandomForestRegressor), "El modelo guardado no es un RandomForestRegressor"
