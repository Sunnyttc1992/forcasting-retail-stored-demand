import streamlit as st
import os
import joblib
import numpy as np

# Configure page
st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")
st.title("📦 Retail Demand Prediction")
st.write("Enter product and promotion details to estimate expected demand based on top 6 features.")

# Load model
model_path = "xgb_top6_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure 'xgb_top6_model.pkl' is present in the app directory.")
    st.stop()
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# Input form matching top 6 features
price = st.slider("Price", 0, 200, 50)
discount = st.slider("Discount (%)", 0, 50, 10)
inventory = st.slider("Inventory Level", 0, 500, 100)
promotion = st.selectbox("Promotion Active?", [0, 1])
units_ordered = st.number_input("Units Ordered", min_value=0, value=200)
epidemic = st.selectbox("Epidemic Impact", [0, 1])

# Predict button
if st.button("Predict Demand"):
    try:
        # Arrange input in the exact order the model expects
        input_data = np.array([[price, discount, inventory, promotion, units_ordered, epidemic]])
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Estimated Demand: **{round(prediction)} units**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
