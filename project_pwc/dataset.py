import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from project_pwc.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def load_dataframes():
    """Carga los CSV en DataFrames separados."""
    salary_df = pd.read_csv(RAW_DATA_DIR / "salary.csv")
    people_df = pd.read_csv(RAW_DATA_DIR / "people.csv")
    desc_df = pd.read_csv(RAW_DATA_DIR / "descriptions.csv")
    return salary_df, people_df, desc_df

def merge_data(salary_df, people_df, desc_df) -> pd.DataFrame:
    """
    Realiza la unificación de people, salary y descriptions en un único DataFrame,
    conservando todos los IDs de people (left join).
    """
    df_merged = people_df.merge(salary_df, on="id", how="left")
    df_merged = df_merged.merge(desc_df, on="id", how="left")
    logger.info(f"Merged data shape: {df_merged.shape}")
    return df_merged

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica estrategias específicas de imputación según el análisis realizado.
    - Salary nulo -> Eliminar filas
    - Description nulo -> "No description"
    - Age nulo -> Mediana
    - Years of Experience nulo -> Mediana
    - Gender, Education, Job Title nulos -> "Missing"
    """
    # Elimina filas con Salary nulo
    before = len(df)
    df = df.dropna(subset=["Salary"])
    after = len(df)
    logger.info(f"Dropped {before - after} rows due to missing Salary.")

    # Descripción nula => "No description"
    df["Description"] = df["Description"].fillna("No description")

    # Age -> rellenar con mediana
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    # Years of Experience -> mediana
    yoe_median = df["Years of Experience"].median()
    df["Years of Experience"] = df["Years of Experience"].fillna(yoe_median)

    # Variables categóricas
    cat_cols = ["Gender", "Education Level", "Job Title"]
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")

    # Convertir Age y Years of Experience a int, si lo deseas
    df["Age"] = df["Age"].astype(int)
    df["Years of Experience"] = df["Years of Experience"].astype(int)

    return df

@app.command()
def main(output_filename: str = "dataset_cleaned.csv"):
    """
    Combina salary, people y descriptions en un solo CSV, imputando los datos faltantes
    con las estrategias acordadas.
    
    Uso:
      python -m project_pwc.dataset --output-filename dataset_cleaned.csv
    """
    logger.info("Starting data cleaning and merging pipeline...")

    # Simulamos proceso con barra de progreso
    for i in tqdm(range(2), total=2):
        pass

    # 1. Carga
    salary_df, people_df, desc_df = load_dataframes()

    # 2. Merge
    df = merge_data(salary_df, people_df, desc_df)

    # 3. Imputación / Limpieza
    df = fill_missing_values(df)

    # 4. Guardar
    output_path = INTERIM_DATA_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset cleaned and saved to {output_path}")

if __name__ == "__main__":
    app()
