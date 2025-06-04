import streamlit as st
import os
import joblib
import numpy as np

st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")
st.title("📦 Retail Demand Prediction")

# Model load with error feedback
if not os.path.exists("xgb_demand_model.pkl"):
    st.error("❌ Model file not found. Please make sure 'xgb_demand_model.pkl' is in the root directory.")
else:
    try:
        model = joblib.load("xgb_demand_model.pkl")
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# Sample inputs (you can customize based on your features)
price = st.slider("Price", 0, 200, 50)
discount = st.slider("Discount", 0, 50, 10)
inventory = st.slider("Inventory Level", 0, 500, 100)
promotion = st.selectbox("Promotion", [0, 1])
units_ordered = st.number_input("Units Ordered", min_value=0, value=200)
epidemic = st.selectbox("Epidemic", [0, 1])

# Add encoded feature placeholders as needed...

# Prediction
if st.button("Predict Demand"):
    input_data = np.array([[price, discount, inventory, promotion, units_ordered, epidemic]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Demand: {round(prediction)} units")
