import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from project_pwc.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def load_dataframes():

    salary_df = pd.read_csv(RAW_DATA_DIR / "salary.csv")
    people_df = pd.read_csv(RAW_DATA_DIR / "people.csv")
    desc_df = pd.read_csv(RAW_DATA_DIR / "descriptions.csv")

    return salary_df, people_df, desc_df

def merge_data(salary_df, people_df, desc_df) -> pd.DataFrame:

    df_merged = people_df.merge(salary_df, on="id", how="left")
    df_merged = df_merged.merge(desc_df, on="id", how="left")
    logger.info(f"Merged data shape: {df_merged.shape}")
    return df_merged

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["Salary"])
    after = len(df)
    logger.info(f"Dropped {before - after} rows due to missing Salary.")

    df["Description"] = df["Description"].fillna("No description")

    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    yoe_median = df["Years of Experience"].median()
    df["Years of Experience"] = df["Years of Experience"].fillna(yoe_median)

    cat_cols = ["Gender", "Education Level", "Job Title"]
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")

    df["Age"] = df["Age"].astype(int)
    df["Years of Experience"] = df["Years of Experience"].astype(int)

    return df

@app.command()
def main(output_filename: str = "dataset_cleaned.csv"):

    logger.info("Starting data cleaning and merging pipeline...")

    for i in tqdm(range(2), total=2):
        pass

    salary_df, people_df, desc_df = load_dataframes()

    df = merge_data(salary_df, people_df, desc_df)

    df = fill_missing_values(df)
    
    output_path = INTERIM_DATA_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset cleaned and saved to {output_path}")

if __name__ == "__main__":
    app()
