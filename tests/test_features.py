import pytest
import pandas as pd
from project_pwc.features import main as features_main
from project_pwc.config import PROCESSED_DATA_DIR

def test_features_no_log():

    features_main(
        input_file="dataset_cleaned.csv", 
        output_file="dataset_features_test.csv",
        use_log_salary=False
    )

    output_path = PROCESSED_DATA_DIR / "dataset_features_test.csv"
    assert output_path.exists(), f"No se generó {output_path}"

    df_test = pd.read_csv(output_path)
    assert "Salary_log" not in df_test.columns, "No deberia existir Salary_log si use_log_salary=False"

def test_features_with_log():

    features_main(
        input_file="dataset_cleaned.csv",
        output_file="dataset_features_test_log.csv",
        use_log_salary=True
    )
    output_path = PROCESSED_DATA_DIR / "dataset_features_test_log.csv"
    assert output_path.exists(), f"No se generó {output_path}"

    df_test = pd.read_csv(output_path)
    assert "Salary_log" in df_test.columns, "Debería crearse 'Salary_log' al usar use_log_salary=True"
