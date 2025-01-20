import pandas as pd
import numpy as np
import typer
from loguru import logger

from project_pwc.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_file: str = "dataset_cleaned.csv",
    output_file: str = "dataset_features.csv",
    use_log_salary: bool = True
) -> None:

    logger.info("Iniciando pipeline")

    input_path = INTERIM_DATA_DIR / input_file
    df = pd.read_csv(input_path)
    logger.info(f"Cargado dataset desde {input_path} con forma {df.shape}")

    if use_log_salary:
        df["Salary_log"] = np.log(df["Salary"])
        logger.info("Aplicada transformación log a 'Salary' -> 'Salary_log'")
    else:
        logger.info("No se aplicó transformación log a 'Salary'.")

    if "Gender" in df.columns:
        df = pd.get_dummies(df, columns=["Gender"], prefix="gender", drop_first=False)
        logger.info("Realizado One-hot encoding de 'Gender'.")

    if "Education Level" in df.columns:
        edu_map = {"Missing": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        df["Education_Level_ordinal"] = df["Education Level"].map(edu_map)
        logger.info("Mapeada 'Education Level' a 'Education_Level_ordinal'.")

    if "Years of Experience" in df.columns:
        bins = [0, 2, 7, 15, float('inf')]
        labels = ["junior", "semi-senior", "senior", "expert"]
        df["experience_level"] = pd.cut(
            df["Years of Experience"], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        logger.info("Creada la columna 'experience_level' mediante pd.cut")

        exp_map = {"junior": 0, "semi-senior": 1, "senior": 2, "expert": 3}
        df["experience_level_ordinal"] = df["experience_level"].map(exp_map)
        logger.info("Mapeada 'experience_level' -> 'experience_level_ordinal'")

    cols_to_drop = [
        "experience_level",
        "id",
        "Description",
        "Job Title",
        "Age",
        "Education Level"
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    logger.info(f"Eliminadas columnas irrelevantes: {cols_to_drop}")

    rename_map = {
        "Years of Experience": "years_experience",
        "gender_Female": "gender_female",
        "gender_Male":   "gender_male"
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    if "gender_Missing" in df.columns:
        df.drop(columns=["gender_Missing"], inplace=True)
        logger.info("Columna 'gender_Missing' eliminada (no se usará en la API).")

    for col in ["gender_female", "gender_male"]:
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(bool)
            logger.info(f"Columna '{col}' convertida a booleano (True/False).")

    output_path = PROCESSED_DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    logger.success(f"Features generadas y guardadas en {output_path}, forma final: {df.shape}")

if __name__ == "__main__":
    app()
