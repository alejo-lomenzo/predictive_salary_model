from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/rf_salary.joblib")

class Features(BaseModel):
    years_experience: float
    gender_female: int
    gender_male: int
    education_level_ordinal: int
    experience_level_ordinal: int

@app.post("/predict")
def predict_salary(data: Features):
    X_input = [[
        data.years_experience,
        data.gender_female,
        data.gender_male,
        data.education_level_ordinal,
        data.experience_level_ordinal
    ]]

    salary_pred = model.predict(X_input)[0]
    return {"predicted_salary": float(salary_pred)}

