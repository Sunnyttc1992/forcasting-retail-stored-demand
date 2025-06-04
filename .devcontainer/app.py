import streamlit as st
import os
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")
st.title("📦 Retail Demand Prediction")
st.write("Enter product and promotion details to estimate expected demand based on selected features.")

# Load model
model_path = "xgb_selected6_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure 'xgb_selected6_model.pkl' is present in the app directory.")
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
weather = st.selectbox("Weather Condition", ['Sunny', 'Snowy', 'Rainy'])  # adjust based on dataset
promotion = st.selectbox("Promotion Active?", [0, 1])
competitor_price = st.slider("Competitor Pricing", 0, 200, 50)
season = st.selectbox("Seasonality", ['Spring', 'Summer', 'Fall', 'Winter'])  # adjust based on dataset

# One-hot encoding (must match model's training)
def encode_input(weather, season):
    weather_map = {'Rainy': [1, 0], 'Snowy': [0, 1], 'Sunny': [0, 0]}
    season_map = {'Spring': [1, 0, 0], 'Summer': [0, 1, 0], 'Winter': [0, 0, 1], 'Fall': [0, 0, 0]}
    return weather_map.get(weather, [0, 0]) + season_map.get(season, [0, 0, 0])

# Assemble input
encoded_weather_season = encode_input(weather, season)
input_data = np.array([[price, discount, promotion, competitor_price] + encoded_weather_season])

# Predict
if st.button("Predict Demand"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Estimated Demand: **{round(prediction)} units**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
