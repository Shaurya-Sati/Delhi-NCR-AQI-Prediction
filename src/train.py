# Model training script — trains XGBoost on weather features
# Full implementation in notebooks/03_models.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# ── Load data ──────────────────────────────
df = pd.read_csv("data/delhi_ncr_aqi_dataset.csv")
print(f"Loaded: {df.shape[0]:,} rows")

# ── Preprocess ─────────────────────────────
df = df.drop(columns=['datetime', 'date', 'station', 'latitude', 'longitude'])

day_map    = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
              'Friday':4,'Saturday':5,'Sunday':6}
season_map = {'monsoon':0,'summer':1,'post_monsoon':2,'winter':3}

df['day_of_week'] = df['day_of_week'].map(day_map)
df['season']      = df['season'].map(season_map)
df['city']        = LabelEncoder().fit_transform(df['city'])

# ── Features & target ──────────────────────
weather_features = ['year','month','day','hour',
                    'day_of_week','is_weekend','season','city',
                    'temperature','humidity','wind_speed','visibility']

X = df[weather_features]
y = df['aqi']

# ── Split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training rows: {X_train.shape[0]:,}")
print(f"Test rows:     {X_test.shape[0]:,}")

# ── Train ──────────────────────────────────
print("Training XGBoost...")
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────
preds  = model.predict(X_test)
mae    = mean_absolute_error(y_test, preds)
r2     = r2_score(y_test, preds)

print(f"Test MAE: {mae:.2f}")
print(f"Test R²:  {r2:.4f}")

# ── Save ───────────────────────────────────
joblib.dump(model, "models/best_model_xgboost.pkl")
print("Model saved to models/best_model_xgboost.pkl")