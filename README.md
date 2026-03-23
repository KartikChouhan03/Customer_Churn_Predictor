# Customer Churn Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg" alt="ML Framework">
  <img src="https://img.shields.io/badge/Web%20App-Streamlit-red.svg" alt="App Framework">
</p>

## 📊 Overview
The **Customer Churn Prediction System** is a Machine Learning application designed to predict whether a bank customer is likely to churn. This project provides end-to-end implementation from exploratory data analysis (EDA) and model training to an interactive web application built with Streamlit.

The model evaluates customer information such as Credit Score, Age, Tenure, Balance, Number of Bank Products, Salary, and Geography to generate a churn prediction along with the churn probability.

## 📂 Project Structure

```text
CustomerChurn_ML/
├── app/
│   └── app.py                  # Streamlit web application frontend
├── data/
│   └── Churn_Modelling.csv     # Raw dataset containing bank customer records
├── models/
│   └── churn_model.pkl         # Trained Machine Learning model (Pickle format)
├── notebooks/
│   └── churn_analysis.ipynb    # Jupyter Notebook with EDA, preprocessing, and training
└── README.md                   # Project documentation
```

## 🚀 Key Features

- **Data Preprocessing & EDA:** Comprehensive exploration of the dataset and feature engineering.
- **Handling Class Imbalance:** Utilization of SMOTE (Synthetic Minority Over-sampling Technique) to address imbalanced data.
- **Model Training & Evaluation:** Training multiple ML algorithms, hyperparameter tuning, and detailed evaluation metrics.
- **Interactive UI:** A simple, easy-to-use Streamlit dashboard for real-time predictions.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/CustomerChurn_ML.git
   cd CustomerChurn_ML
   ```

2. **Install the dependencies:**
   Make sure you have Python installed. You can install the required packages using:
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```

3. **Run the Streamlit Application:**
   ```bash
   streamlit run app/app.py
   ```
   *The application will automatically open in your default web browser.*

## 🧠 Model Pipeline

The machine learning pipeline included within `notebooks/churn_analysis.ipynb` incorporates:
1. Data Preprocessing & Cleaning
2. Exploratory Data Analysis (EDA)
3. Handling target class imbalance via SMOTE
4. Feature Scaling (Standardization/Normalization)
5. Model Building using multiple classifiers (e.g., Logistic Regression, Random Forest, XGBoost)
6. Hyperparameter Tuning for optimized predictive performance
7. Model Evaluation and final model selection

## 📋 App Usage

On the Streamlit dashboard, enter the following customer details to get a churn prediction:
- **Demographics:** Age, Gender, Geography
- **Account Info:** Credit Score, Tenure, Account Balance, Estimated Salary
- **Bank Products:** Number of Bank Products, Has Credit Card, Is Active Member

After setting the values using the provided sliders and inputs, click **"Predict Churn"** to see if the customer is likely to stay or leave, along with the calculated probability.

## 🤝 Contributing
Contributions are always welcome. Feel free to fork this repository, make enhancements, and submit a pull request!


