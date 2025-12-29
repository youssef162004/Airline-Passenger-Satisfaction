import streamlit as st
import numpy as np
import joblib
import pandas as pd

# تحميل الملفات
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.title("✈️ Airline Passenger Satisfaction (Logistic Regression)")

st.write("Enter passenger information")

# ===== Inputs =====
age = st.slider("Age", 7, 85, 30)
flight_distance = st.number_input("Flight Distance", 0, 5000, 1000)
departure_delay = st.number_input("Departure Delay", 0, 300, 0)
arrival_delay = st.number_input("Arrival Delay", 0, 300, 0)

gender = st.selectbox("Gender", ["Female", "Male"])
customer_type = st.selectbox("Customer Type", ["First-time", "Returning"])
travel_type = st.selectbox("Type of Travel", ["Personal", "Business"])
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])

# ===== Create empty input =====
input_dict = dict.fromkeys(features, 0)

# ===== Fill numeric values =====
input_dict['Age'] = age
input_dict['Flight Distance'] = flight_distance
input_dict['Departure Delay'] = departure_delay
input_dict['Arrival Delay'] = arrival_delay

# ===== One-Hot Encoding (نفس التدريب) =====
if gender == "Male":
    input_dict['Gender_Male'] = 1

if customer_type == "Returning":
    input_dict['Customer Type_Returning'] = 1

if travel_type == "Business":
    input_dict['Type of Travel_Business'] = 1

if travel_class == "Eco Plus":
    input_dict['Class_Eco Plus'] = 1
elif travel_class == "Business":
    input_dict['Class_Business'] = 1

# ===== DataFrame =====
input_df = pd.DataFrame([input_dict])

# ===== Scaling =====
input_scaled = scaler.transform(input_df)

# ===== Prediction =====
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"😊 Passenger is SATISFIED (Probability: {probability:.2f})")
    else:
        st.error(f"😞 Passenger is NOT satisfied (Probability: {probability:.2f})")
