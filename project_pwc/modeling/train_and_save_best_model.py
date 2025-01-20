import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib
from loguru import logger

from project_pwc.config import PROCESSED_DATA_DIR

def train_and_save_model(
    input_csv= (PROCESSED_DATA_DIR / 'dataset_features.csv'),
    output_model="models/rf_salary.joblib"
):

    df = pd.read_csv(input_csv)

    X = df.drop(columns=["Salary","Salary_log"], errors="ignore")
    y = df["Salary"]

    logger.info(f"Dataset shape: {df.shape}. Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_best = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    logger.info("Entrenando RandomForest con los hiperparámetros óptimos...")
    rf_best.fit(X_train, y_train)

    y_pred = rf_best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    logger.info(f"MAE en test: {mae:.2f}")
    logger.info(f"R² en test:  {r2:.3f}")

    logger.info(f"Guardando modelo en {output_model}")
    joblib.dump(rf_best, output_model)
    logger.success("Modelo guardado exitosamente!")

if __name__ == "__main__":
    train_and_save_model()
