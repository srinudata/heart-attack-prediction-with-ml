import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("heart_attack_prediction_randomforest.pkl", "rb"))

st.title("❤️ Heart Attack Prediction")

with st.form("heart_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["Urban", "Rural"])
    income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    obesity = st.selectbox("Obesity", [0, 1])
    waist_circumference = st.number_input("Waist Circumference (cm)", min_value=50, max_value=200, value=90)
    family_history = st.selectbox("Family History of Heart Disease", [0, 1])
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
    physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
    air_pollution_exposure = st.selectbox("Air Pollution Exposure", ["Low", "Medium", "High"])
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    blood_pressure_systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
    blood_pressure_diastolic = st.number_input("Diastolic BP", min_value=50, max_value=120, value=80)
    fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50, max_value=300, value=100)
    cholesterol_hdl = st.number_input("HDL Cholesterol", min_value=20, max_value=100, value=50)
    cholesterol_ldl = st.number_input("LDL Cholesterol", min_value=50, max_value=300, value=120)
    triglycerides = st.number_input("Triglycerides", min_value=50, max_value=500, value=150)
    ekg_results = st.selectbox("EKG Results", ["Normal", "Abnormal"])
    previous_heart_disease = st.selectbox("Previous Heart Disease", [0, 1])
    medication_usage = st.selectbox("Medication Usage", [0, 1])
    participated_in_free_screening = st.selectbox("Participated in Free Screening", [0, 1])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_data = {
        'age': [age],
        'gender': [gender],
        'region': [region],
        'income_level': [income_level],
        'hypertension': [hypertension],
        'diabetes': [diabetes],
        'cholesterol_level': [cholesterol_level],
        'obesity': [obesity],
        'waist_circumference': [waist_circumference],
        'family_history': [family_history],
        'smoking_status': [smoking_status],
        'alcohol_consumption': [alcohol_consumption],
        'physical_activity': [physical_activity],
        'dietary_habits': [dietary_habits],
        'air_pollution_exposure': [air_pollution_exposure],
        'stress_level': [stress_level],
        'sleep_hours': [sleep_hours],
        'blood_pressure_systolic': [blood_pressure_systolic],
        'blood_pressure_diastolic': [blood_pressure_diastolic],
        'fasting_blood_sugar': [fasting_blood_sugar],
        'cholesterol_hdl': [cholesterol_hdl],
        'cholesterol_ldl': [cholesterol_ldl],
        'triglycerides': [triglycerides],
        'EKG_results': [ekg_results],
        'previous_heart_disease': [previous_heart_disease],
        'medication_usage': [medication_usage],
        'participated_in_free_screening': [participated_in_free_screening]
    }

    input_df = pd.DataFrame(user_data)

    input_df = input_df.replace({None: np.nan, "": np.nan})

    input_df = input_df.astype("object")
    for col in input_df.columns:
        if isinstance(input_df[col].iloc[0], (int, float, np.integer, np.floating)):
            input_df[col] = input_df[col].astype(float)

    st.write("✅ Input Data:")
    st.dataframe(input_df)
    st.write("🔎 Data types:", input_df.dtypes)

    prediction = model.predict(input_df)
    st.subheader("📊 Prediction Result")
    st.write("Prediction:", prediction[0])
    