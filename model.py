# -----------------------------------------------------------
# File: model.py
# Description: Train and save Random Forest Regressor for AQI
# Dataset: city_day.csv (from Kaggle or OpenAQ)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# ✅ Step 1: Load Dataset
try:
    data = pd.read_csv("city_day.csv")
    print("✅ city_day.csv loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError("❌ city_day.csv not found. Please place it in the same folder as this script.")

# ✅ Step 2: Keep only useful columns
expected_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
data = data[expected_cols]

# ✅ Step 3: Handle missing values
data = data.dropna()

# ✅ Step 4: Features and target
X = data.drop('AQI', axis=1)
y = data['AQI']

# ✅ Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 7: Save model
with open('aqi_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained successfully and saved as aqi_model.pkl")
print(f"📊 Dataset used: {data.shape[0]} rows, {data.shape[1]} columns")
