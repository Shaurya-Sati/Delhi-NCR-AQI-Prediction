# Prediction script — loads saved model and predicts AQI for new input
# Full implementation in notebooks/04_evaluation.ipynb
import joblib
import numpy as np

# ── Load saved model ───────────────────────
model = joblib.load("models/best_model_xgboost.pkl")

# ── Input — change these values to predict ─
year       = 2024
month      = 12      # December
day        = 15
hour       = 8       # 8am
day_of_week = 6      # Sunday
is_weekend = 1
season     = 3       # winter
city       = 0       # Delhi
temperature = 10.0
humidity   = 85
wind_speed = 2.5
visibility = 3.0

# ── Predict ────────────────────────────────
features = np.array([[year, month, day, hour,
                      day_of_week, is_weekend, season, city,
                      temperature, humidity, wind_speed, visibility]])

prediction = model.predict(features)[0]

print(f"Predicted AQI: {prediction:.0f}")

if prediction <= 50:    category = "Good"
elif prediction <= 100: category = "Satisfactory"
elif prediction <= 200: category = "Moderate"
elif prediction <= 300: category = "Poor"
elif prediction <= 400: category = "Very Poor"
else:                   category = "Severe"

print(f"Category:      {category}")