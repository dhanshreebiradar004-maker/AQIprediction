# -----------------------------------------------------------
# File: app.py
# Description: Streamlit Web App for AQI Prediction
# -----------------------------------------------------------

import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('aqi_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Air Quality Index Predictor", layout="centered")

# --- UI Design ---
st.markdown("<h1 style='text-align:center;'>🌫️ Air Quality Index Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter pollutant concentrations to predict the Air Quality Index</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Pollutant Concentrations")
st.caption("Input the measured values for each pollutant")

# Input fields in two-column layout
col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0)
    no = st.number_input("NO (µg/m³)", min_value=0.0)
    nox = st.number_input("NOx (ppb)", min_value=0.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0)
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0)
    toluene = st.number_input("Toluene (µg/m³)", min_value=0.0)
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0)
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0)
    benzene = st.number_input("Benzene (µg/m³)", min_value=0.0)
    xylene = st.number_input("Xylene (µg/m³)", min_value=0.0)

# Prediction button
if st.button("Predict Air Quality Index"):
    # Prepare data
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]])
    prediction = model.predict(input_data)
    aqi = round(prediction[0], 2)

    st.success(f"🌍 Predicted Air Quality Index (AQI): **{aqi}**")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Powered by Random Forest Regression")
