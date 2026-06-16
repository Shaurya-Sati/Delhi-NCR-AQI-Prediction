import streamlit as st
import joblib
import numpy as np

# ── Page config ────────────────────────────
st.set_page_config(
    page_title="Delhi NCR AQI Predictor",
    page_icon="🌫️",
    layout="centered"
)

# ── Load model ─────────────────────────────
model = joblib.load("models/best_model_xgboost.pkl")

# ── Title ──────────────────────────────────
st.title("🌫️ Delhi NCR AQI Predictor")
st.markdown("Predict Air Quality Index using weather and time features.")
st.divider()

# ── Inputs ─────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", 
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: ['Delhi','Faridabad','Ghaziabad','Gurugram','Noida'][x])
    
    season = st.selectbox("Season",
        options=[0, 1, 2, 3],
        format_func=lambda x: ['Monsoon','Summer','Post-Monsoon','Winter'][x])
    
    month = st.slider("Month", 1, 12, 6)
    hour  = st.slider("Hour of day", 0, 23, 8)

with col2:
    temperature = st.slider("Temperature (°C)", -5, 48, 20)
    humidity    = st.slider("Humidity (%)", 10, 100, 60)
    wind_speed  = st.slider("Wind Speed (km/h)", 0.0, 30.0, 5.0)
    visibility  = st.slider("Visibility (km)", 0.0, 20.0, 8.0)

st.divider()

# ── Predict button ─────────────────────────
if st.button("🔍 Predict AQI", use_container_width=True):
    
    day_of_week = 2
    is_weekend  = 1 if day_of_week >= 5 else 0
    day         = 15
    year        = 2024

    features = np.array([[year, month, day, hour,
                          day_of_week, is_weekend, season, city,
                          temperature, humidity, wind_speed, visibility]])

    prediction = model.predict(features)[0]

    if prediction <= 50:    category, color = "Good", "🟢"
    elif prediction <= 100: category, color = "Satisfactory", "🟡"
    elif prediction <= 200: category, color = "Moderate", "🟠"
    elif prediction <= 300: category, color = "Poor", "🔴"
    elif prediction <= 400: category, color = "Very Poor", "🔴"
    else:                   category, color = "Severe", "🚨"

    st.metric("Predicted AQI", f"{prediction:.0f}")
    st.markdown(f"## {color} {category}")