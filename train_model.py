# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:41:02 2025

@author: hp
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
file_path = r"C:\Users\hp\Desktop\train_data.csv"  # Update this with the actual file path
df = pd.read_csv(file_path)

# Drop Loan_ID column (not useful for model training)
df.drop(columns=["Loan_ID"], inplace=True)

# Fix 'Dependents' column
df["Dependents"] = df["Dependents"].replace("3+", 3)  # Convert '3+' to 3
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0]).astype(int)  # Replace NaN with mode

# Convert categorical values to numerical
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})  # Convert target variable to 0 & 1

# Handle missing values in numerical columns
df.fillna(df.mean(), inplace=True)

# Define Features (X) and Target Variable (y)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model to a File
joblib.dump(model, "loan_model.pkl")

print("✅ Model trained and saved successfully as 'loan_model.pkl'!")

