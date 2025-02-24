# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:49:09 2025

@author: hp
"""

import pandas as pd
import joblib
import numpy as np

# Load test dataset
test_df = pd.read_csv(r"C:\Users\hp\Desktop\test_data.csv")  # Change path if needed

# Save Loan_IDs for final output
loan_ids = test_df["Loan_ID"]

# Drop Loan_ID column
test_df.drop(columns=["Loan_ID"], inplace=True)

# Fix 'Dependents' column
test_df["Dependents"] = test_df["Dependents"].replace("3+", 3)
test_df["Dependents"] = test_df["Dependents"].fillna(test_df["Dependents"].mode()[0]).astype(int)

# Convert categorical values to numerical
test_df["Gender"] = test_df["Gender"].map({"Male": 1, "Female": 0})
test_df["Married"] = test_df["Married"].map({"Yes": 1, "No": 0})
test_df["Education"] = test_df["Education"].map({"Graduate": 1, "Not Graduate": 0})
test_df["Self_Employed"] = test_df["Self_Employed"].map({"Yes": 1, "No": 0})
test_df["Property_Area"] = test_df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})

# Handle missing values
test_df.fillna(test_df.mean(), inplace=True)

# Load trained model
model = joblib.load("loan_model.pkl")

# Make predictions
predictions = model.predict(test_df)

# Convert predictions back to 'Y' and 'N'
predictions = ["Y" if pred == 1 else "N" for pred in predictions]

# Save results to a CSV file
output_df = pd.DataFrame({"Loan_ID": loan_ids, "Loan_Status": predictions})
output_df.to_csv("loan_predictions.csv", index=False)

print("Predictions saved to loan_predictions.csv successfully!")
