import streamlit as st
import os
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")
st.title("📦 Retail Demand Prediction")
st.write("Enter product and promotion details to estimate expected demand.")

# 🔍 Load model
model_path = "xgb_demand_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure 'xgb_demand_model.pkl' exists in the root directory.")
    st.stop()
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# 📥 Input features (customize according to your model's training features)
price = st.slider("Price", 0, 200, 50)
discount = st.slider("Discount (%)", 0, 50, 10)
inventory = st.slider("Inventory Level", 0, 500, 100)
promotion = st.selectbox("Promotion Active?", [0, 1])
units_ordered = st.number_input("Units Ordered", min_value=0, value=200)
epidemic = st.selectbox("Epidemic Impact", [0, 1])

# 🧠 Predict demand
if st.button("Predict Demand"):
    try:
        # Adjust feature order to match your training data
        input_data = np.array([[price, discount, inventory, promotion, units_ordered, epidemic]])
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Estimated Demand: **{round(prediction)} units**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
