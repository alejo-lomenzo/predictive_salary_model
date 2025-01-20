import numpy as np
import joblib
import streamlit as st

def run_streamlit_app():

    st.title("Salary Prediction (Without Log)")

    model = joblib.load("models/rf_salary.joblib")

    years_experience = st.number_input(
        label="Years of Experience",
        min_value=0,
        max_value=40,
        value=5,
        help="Número total de años de experiencia."
    )

    gender = st.selectbox(
        label="Gender",
        options=["Male", "Female"],
        index=0
    )

    edu_level = st.selectbox(
        label="Education Level",
        options=["Missing", "Bachelor's", "Master's", "PhD"],
        index=1
    )
    edu_map = {"Missing":0,"Bachelor's":1,"Master's":2,"PhD":3}
    edu_ordinal = edu_map[edu_level]

    exp_level_map = {"junior":0, "semi-senior":1, "senior":2, "expert":3}
    exp_level_choice = st.selectbox(
        label="Experience Level Ordinal",
        options=["junior","semi-senior","senior","expert"],
        index=0
    )
    experience_level_ordinal = exp_level_map[exp_level_choice]

    gender_female = 0
    gender_male   = 0
    if gender == "Male":
        gender_male = 1
    elif gender == "Female":
        gender_female = 1

    if st.button("Predict"):
        X_input = np.array([[
            years_experience,      
            gender_female,        
            gender_male,           
            edu_ordinal,          
            experience_level_ordinal  
        ]], dtype=float)

        salary_pred = model.predict(X_input)[0]

        st.write(f"Predicted Salary: ${salary_pred:,.2f}")

if __name__ == "__main__":
    run_streamlit_app()