# project_pwc

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
  <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains the **end-to-end pipeline** for **predicting salaries** based on multiple features (Years of Experience, Gender, Education Level, etc.). It covers data loading and cleaning, Exploratory Data Analysis (EDA), model training (RandomForest, LinearRegression, Ensemble), advanced validation techniques (NestedCV, RepeatedKFold), and a simple **API** (FastAPI) plus **UI** (Streamlit) for real-time inference. It also includes Pytest scripts covering each step of the pipeline.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Data Exploration and Cleaning](#data-exploration-and-cleaning)  
3. [Advanced EDA](#advanced-eda)  
4. [Model Training](#model-training)  
5. [Advanced Validation and Ensemble](#advanced-validation-and-ensemble)  
6. [API and UI Deployment](#api-and-ui-deployment)  
7. [Testing with Pytest](#testing-with-pytest)  
8. [Overall Conclusions](#overall-conclusions)

---

<details> <summary><strong>1. Project Structure</strong></summary>

├── LICENSE
├── Makefile
├── README.md                          <- Main documentation
├── data
│   ├── external                       <- External/third-party data
│   ├── interim                        <- Intermediate data (partially cleaned)
│   ├── processed                      <- Final, ready-to-model datasets
│   └── raw                            <- Original, immutable data
│
├── docs                               <- Documentation (e.g., mkdocs)
│
├── models                             <- Trained models (.joblib) and model outputs
│
├── notebooks
│   ├── 1.0-eda-initial-exploration.ipynb
│   ├── 2.0-eda-advanced-exploration.ipynb
│   ├── 3.0-model-training.ipynb
│   ├── 4.0-explain-linear-assumptions.ipynb
│   ├── 5.0-hyperparam-tuning-grid.ipynb
│   ├── 5.1-hyperparam-optuna.ipynb
│   ├── 6.0-advanced-validation.ipynb
│   └── 7.0-ensemble.ipynb
│
├── pyproject.toml                     <- Project config (formatters, linters)
├── references                         <- Manuals, data dictionaries, etc.
├── reports                            <- HTML/PDF analyses
│   └── figures                        <- Figures/plots used in reporting
├── requirements.txt                   <- Frozen pip dependencies
├── setup.cfg                          <- flake8 config
├── tests
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_modeling.py
│   └── test_api.py
└── project_pwc
    ├── __init__.py
    ├── config.py
    ├── dataset.py                     <- Data loading and cleaning
    ├── features.py                    <- Feature engineering
    ├── modeling
    │   ├── __init__.py
    │   ├── train_and_save_best_model.py
    │   └── ...
    ├── api
    │   └── app.py                     <- FastAPI endpoint
    └── ui
        └── app.py                     <- Streamlit UI
</details>

---

## 2. Data Exploration and Cleaning <a id="data-exploration-and-cleaning"></a>

### **dataset.py**

**Goal**:
- **Load** three raw CSVs: `salary.csv`, `people.csv`, `descriptions.csv`.
- **Merge** them on `id`.
- **Fill or remove** missing values:
  - Drop rows with no salary (target missing).
  - Fill `Description` with `"No description"`.
  - Fill `Age` / `Years of Experience` with median, convert to `int`.
  - Any categorical columns (`Gender`, `Education`, etc.) missing are replaced with `"Missing"`.
- **Save** the result as `dataset_cleaned.csv` in `data/interim/`.

**Usage**:
python -m project_pwc.dataset

Output: data/interim/dataset_cleaned.csv

In Notebooks:
1.0-eda-initial-exploration.ipynb does a preliminary check on these CSVs (shape, columns).
2.0-eda-advanced-exploration.ipynb goes deeper (correlation, outliers, distributions).

## 3. Advanced EDA <a id="advanced-eda"></a>
1.0-eda-initial-exploration
Loads raw data, merges them to view potential missing columns and general shape.
2.0-eda-advanced-exploration
Correlation Heatmap: Age ~ YearsOfExperience (~0.98), both ~0.92–0.93 to Salary → potential multicollinearity for linear models.
Boxplots & Histograms:
Salary shows a strong right-skew (long tail up to 200k+).
Age is roughly normal (23–53).
Experience peaks at ~0–5 and ~15 years.
Categorical vs. Salary:
Education Level (Bachelor’s < Master’s < PhD) correlates with higher salaries.
Gender indicates differences in salary distribution.
Scatter (Age vs. Experience vs. Salary) → outliers, strong collinearity, possible log transformation for Salary in linear models.

## 4. Model Training <a id="model-training"></a>
3.0-model-training
Features:

Optionally create df["Salary_log"].
One-hot encode Gender.
Ordinal-encode Education.
Bin YearsOfExperience → experience_level_ordinal.
Models:

DummyRegressor (baseline): MAE ~$40k, R² ~ -0.00
LinearRegression: ~$10.8k MAE, ~0.91 R² with direct Salary
RandomForestRegressor: ~$10k MAE, ~0.88–0.91 R²
If using Salary_log, the model can reach ~0.92 in log-scale R² but ~0.88 when reverting to real scale.
Conclusion:

RandomForest is best among these basic models.
The difference between direct Salary vs. log-scale is not large in final dollar MAE.

## 5. Advanced Validation and Ensemble <a id="advanced-validation-and-ensemble"></a
4.0-explain-linear-assumptions
Checks linear model assumptions via Residual vs. Predicted and Q-Q Plot.
Residuals are fairly normal except for heavier tails (salary outliers).

5.0-hyperparam-tuning-grid
GridSearchCV on RandomForest (Salary).
Best params: e.g., max_depth=None, min_samples_split=5, n_estimators=200, with MAE ~$10.5k, R² ~0.88.

5.1-hyperparam-optuna
Optuna for a broader hyperparameter search.
Sometimes sees extremely low test MAE (~$600), but the average remains near $10k.
Illustrates that advanced searches can yield specialized solutions or rely on favorable splits.

6.0-advanced-validation
RepeatedKFold: MSE ~2.51e8 ± 6.89e7 → RMSE ~$15.8k ± $8k.
NestedCV: MAE ~$10.3k ± $722.
Confirms the model’s average error stays around $10k–$11k across various folds.

7.0-ensemble
A VotingRegressor combining RandomForest, XGBRegressor, and LinearRegression.
Results: MAE ~$10.4k, R²=0.90, very similar to a well-tuned RandomForest alone.

## 6. API and UI Deployment <a id="api-and-ui-deployment"></a>
API: project_pwc/api/app.py
Loads models/rf_salary.joblib.
Exposes POST /predict (FastAPI) with a Pydantic schema:
{
  "years_experience": 5,
  "gender_female": 1,
  "gender_male": 0,
  "education_level_ordinal": 2,
  "experience_level_ordinal": 1
}
Returns {"predicted_salary": <float>}.

Usage:
uvicorn project_pwc.api.app:app --host 0.0.0.0 --port 8000

Send test requests via Postman or curl:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"years_experience":5, "gender_female":1, "gender_male":0, ...}'

UI: project_pwc/ui/app.py (Streamlit)

A simple form to input:
Years of Experience, Gender, Education Level, Experience Level Ordinal.
Runs model.predict(...) and displays the estimated salary.

Usage:
streamlit run project_pwc/ui/app.py
Opens at http://localhost:8501.

## 7. Testing with Pytest <a id="testing-pytest"></a>
The tests/ folder includes several scripts to ensure the pipeline works as expected:

test_dataset.py

test_load_dataframes: Checks load_dataframes() returns non-empty DataFrames.
test_merge_data: Ensures merged DataFrame has Salary, Description.
test_fill_missing_values: Validates no more nulls in Salary and that Age/Experience are int.
test_features.py

test_features_no_log: Calls features.main(..., use_log_salary=False), verifying Salary_log is NOT created.
test_features_with_log: Calls the same with True, checks Salary_log is indeed created.
test_modeling.py

test_train_and_save_model: Runs train_and_save_best_model on a sample CSV, confirms it outputs a .joblib file and that it is a RandomForestRegressor.
test_api.py

test_predict_salary: Uses FastAPI’s TestClient to POST a sample JSON to /predict and expects status_code == 200 and a predicted_salary field.
How to run:
pipenv shell
pytest tests/

All tests should pass, possibly showing minor warnings (Pandas “SettingWithCopyWarning” or scikit-learn “X does not have valid feature names”).

## 8. Overall Conclusions <a id="overall-conclusions"></a>
Model Performance:

A well-tuned RandomForest yields ~$10k MAE (5% error if Salary can reach 200k).
LinearRegression is simpler but has a higher MAE ($10.8k). Baseline is far worse ($40k).
Using Salary_log or direct Salary produce similar results when evaluated in real currency.
Validation:

RepeatedKFold and NestedCV confirm that ~$10k–$11k error is stable across splits.
Ensemble (Voting) slightly improves or matches the best RandomForest (MAE ~$10.4k).
API and UI:

The final pipeline is exposed via a FastAPI endpoint for programmatic requests.
A Streamlit interface allows direct, user-friendly input of features and immediate predictions.
Testing:

The Pytest suite ensures every phase (data merging, feature engineering, modeling, and API) is tested and functioning.
All tests pass, indicating a consistent pipeline.
Future Work:

Handling outliers more explicitly or advanced interpretability (e.g., SHAP, PDP).
Possibly exploring Docker for containerized deployment.
Integrating more hyperparam search or sophisticated ensembles if further error reduction is needed.
Overall, the project demonstrates a robust pipeline that predicts salary with an MAE of ~$10k and R² near 0.90. The validations confirm stability, and the tested API/UI provide easy access.