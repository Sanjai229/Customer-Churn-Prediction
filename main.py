# main.py
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Columns exactly as used during training
model_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                  'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                  'MultipleLines_No phone service', 'MultipleLines_Yes',
                  'InternetService_Fiber optic', 'InternetService_No',
                  'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                  'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                  'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                  'TechSupport_No internet service', 'TechSupport_Yes',
                  'StreamingTV_No internet service', 'StreamingTV_Yes',
                  'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                  'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                  'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                  'PaymentMethod_Mailed check']

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Create DataFrame with all zeros for all model features
        df = pd.DataFrame([{col: 0 for col in model_features}])

        # Fill numeric columns from input
        for col in numeric_cols:
            if col in data:
                df[col] = data[col]

        # Scale numeric columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Fill categorical columns from input
        for col in model_features:
            if col in data:
                df[col] = 1

        # Ensure column order matches model_features
        df = df[model_features]

        # Make prediction
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
