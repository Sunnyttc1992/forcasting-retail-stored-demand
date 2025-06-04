import streamlit as st
import os
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")
st.title("📦 Retail Demand Prediction")
st.write("Enter product and promotion details to estimate expected demand based on selected features.")

# Load model
model_path = "xgb_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure 'xgb_model.pkl' is present in the app directory.")
    st.stop()
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# Inputs matching final selected features
price = st.slider("Price", 0, 200, 50)
discount = st.slider("Discount (%)", 0, 50, 10)
weather = st.selectbox("Weather Condition", ['Snowy', 'Cloudy', 'Sunny', 'Rainy'])  # adjust based on dataset
promotion = st.selectbox("Promotion Active?", [0, 1])
competitor_price = st.slider("Competitor Pricing", 0, 200, 50)
season = st.selectbox("Seasonality", ['Winter', 'Spring', 'Summer', 'Autumn'])  # adjust based on dataset

# One-hot encoding (must match model's training)
def encode_input(weather, season):
    # One-hot for Weather Condition (drop_first=True assumed: 'Cloudy' is dropped)
    weather_map = {
        'Snowy': [1, 0, 0],
        'Sunny': [0, 1, 0],
        'Rainy': [0, 0, 1],
        'Cloudy': [0, 0, 0]  # base case
    }

    # One-hot for Seasonality (drop_first=True assumed: 'Autumn' is dropped)
    season_map = {
        'Winter': [1, 0, 0],
        'Spring': [0, 1, 0],
        'Summer': [0, 0, 1],
        'Autumn': [0, 0, 0]  # base case
    }

    return weather_map.get(weather, [0, 0, 0]) + season_map.get(season, [0, 0, 0])

# Assemble input
encoded_features = encode_input(weather, season)

# Add any additional numeric input here if used in training!
input_data = np.array([[price, discount, promotion, competitor_price] + encoded_features])

# Predict
if st.button("Predict Demand"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Estimated Demand: **{round(prediction)} units**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
