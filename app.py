import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.write("Predict whether a patient is diabetic using Logistic Regression.")

st.sidebar.header("Enter Patient Details")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 250, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 79)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 30.5)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47)
age = st.sidebar.number_input("Age", 1, 120, 33)

input_data = np.array([[preg,
                        glucose,
                        bp,
                        skin,
                        insulin,
                        bmi,
                        dpf,
                        age]])

if st.button("Predict"):

    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("⚠️ The patient is likely Diabetic.")
    else:
        st.success("✅ The patient is likely Non-Diabetic.")

    st.write(f"### Probability of Diabetes: **{probability:.2%}**")
