import pandas as pd
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from project_pwc.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_file: str = "dataset_cleaned.csv",
    output_file: str = "dataset_features.csv",
    use_log_salary: bool = True
):

    logger.info("Iniciando pipeline de feature engineering...")

    input_path = INTERIM_DATA_DIR / input_file
    df = pd.read_csv(input_path)
    logger.info(f"Cargado dataset desde {input_path} con forma {df.shape}")

    if use_log_salary:
        df["Salary_log"] = np.log(df["Salary"])
        logger.info("Se aplicó log a la columna Salary -> Salary_log")
    else:
        logger.info("No se aplicó transformación log a Salary.")

    cols_to_drop = ["id", "Description", "Age", "Job Title", "Salary"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    logger.info(f"Se eliminaron columnas irrelevantes: {cols_to_drop}")

    if "Gender" in df.columns:
        df = pd.get_dummies(df, columns=["Gender"], prefix="gender", drop_first=False)
        logger.info("Realizado One-hot encoding de Gender.")

    if "Education Level" in df.columns:
        edu_map = {
            "Missing": 0,
            "Bachelor's": 1,
            "Master's": 2,
            "PhD": 3
        }
        df["Education_Level_ordinal"] = df["Education Level"].map(edu_map)
        df.drop(columns=["Education Level"], inplace=True)
        logger.info("Realizado Ordinal encoding de Education Level.")

    if "Years of Experience" in df.columns:
        bins = [0, 2, 7, 15, float('inf')]
        labels = [0, 1, 2, 3]  # 0=junior, 1=semi-senior, 2=senior, 3=expert
        df["experience_level_ordinal"] = pd.cut(
            df["Years of Experience"], bins=bins, labels=labels, include_lowest=True
        ).astype(int)

        
        logger.info("Se creó experience_level_ordinal a partir de Years of Experience.")

    output_path = PROCESSED_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.success(f"Features generadas y guardadas en {output_path}, forma final: {df.shape}")

if __name__ == "__main__":
    app()
