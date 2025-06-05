🛒 Forecasting Retail Store Demand
This project demonstrates how to apply a machine learning model to predict product demand in a retail store environment.

🔧 Tools & Technologies Used
Python 🐍 – core programming language

Pandas – for data wrangling and analysis

Matplotlib / Seaborn – for exploratory data analysis and visualizations

Scikit-learn – for preprocessing and model evaluation

XGBoost – for training a high-performance regression model

Streamlit – for building an interactive web app interface

Jupyter Notebook – for development and documentation

📈 Model Overview
The model is trained using the following 7 input features:

Price

Discount

Promotion

Competitor Pricing

Units Sold

Weather Condition (one-hot encoded)

Seasonality (one-hot encoded)

The model of choice is XGBoost Regressor, a powerful gradient boosting framework known for its accuracy and efficiency.

🧪 Performance Metrics
After evaluation on a holdout test set, the model achieved:

✅ RMSE: 24.05

✅ R² Score: 0.737

🌐 Deployment
The final model is deployed through an interactive Streamlit application, allowing users to input real-time values for the 7 features and receive instant demand predictions.


