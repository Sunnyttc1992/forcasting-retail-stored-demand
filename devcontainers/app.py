import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("xgb_demand_model.pkl")

st.title("📦 Demand Prediction App")
st.write("Enter product details to predict demand")

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
