import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write("Predict whether a bank customer is likely to churn.")

# ----------------------------
# Load Model
# ----------------------------

model = pickle.load(open("../models/churn_model.pkl", "rb"))
# ----------------------------
# Input Section
# ----------------------------

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.slider("Credit Score", 300, 900, 650)

    age = st.slider("Age", 18, 100, 35)

    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)

    balance = st.number_input("Account Balance", value=50000.0)

    num_products = st.slider("Number of Bank Products", 1, 4, 1)


with col2:

    has_card = st.selectbox("Has Credit Card", [0,1])

    active_member = st.selectbox("Is Active Member", [0,1])

    salary = st.number_input("Estimated Salary", value=50000.0)

    geography = st.selectbox("Geography", ["France","Germany","Spain"])

    gender = st.selectbox("Gender", ["Male","Female"])

# ----------------------------
# Encoding Categorical Values
# ----------------------------

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

# ----------------------------
# Feature Vector
# ----------------------------

features = np.array([[

credit_score,
age,
tenure,
balance,
num_products,
has_card,
active_member,
salary,
geo_germany,
geo_spain,
gender_male

]])

# ----------------------------
# Prediction
# ----------------------------

st.header("Prediction")

if st.button("Predict Churn"):

    prediction = model.predict(features)

    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer likely to churn")
        st.write(f"Churn Probability: **{probability:.2f}**")

    else:
        st.success("Customer likely to stay")
        st.write(f"Churn Probability: **{probability:.2f}**")

# ----------------------------
# Information Section
# ----------------------------

st.markdown("---")

st.subheader("About This Model")

st.write("""
This system predicts customer churn using a Machine Learning model trained on banking customer data.

The pipeline included:

- Data preprocessing
- Exploratory Data Analysis
- SMOTE for class imbalance
- Feature scaling
- Multiple ML models
- Hyperparameter tuning
- Model evaluation
""")

