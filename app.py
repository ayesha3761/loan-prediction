# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:40:30 2025

@author: hp
"""

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load(r"C:\Users\hp\Desktop\loan prediction\loan_model.pkl")

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")

# Collect user input
gender = st.radio("Gender", ("Male", "Female"))
married = st.radio("Married", ("Yes", "No"))
education = st.radio("Education", ("Graduate", "Not Graduate"))
self_employed = st.radio("Self Employed", ("Yes", "No"))
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Term (in months)", [12, 36, 60, 120, 180, 240, 300, 360])
credit_history = st.radio("Credit History", ("Yes", "No"))
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert user input to numerical format
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
credit_history = 1 if credit_history == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Create input array
input_data = np.array([[gender, married, education, self_employed, dependents, 
                        applicant_income, coapplicant_income, loan_amount, loan_term, 
                        credit_history, property_area]])

# Predict loan approval
if st.button("Check Loan Eligibility"):
    prediction = model.predict(input_data)
    result = "✅ Loan Approved!" if prediction[0] == 1 else "❌ Loan Not Approved"
    st.success(result)
