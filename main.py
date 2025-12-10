from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# All model features after preprocessing (numeric + one-hot)
model_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                  'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                  'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic',
                  'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                  'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service',
                  'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes',
                  'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service',
                  'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                  'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                  'PaymentMethod_Mailed check']

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Scale only numeric columns
        for col in numeric_cols:
            if col in input_df:
                input_df[[col]] = scaler.transform(input_df[[col]])

        # One-hot encode categorical columns
        input_encoded = pd.get_dummies(input_df)

        # Add missing columns from model_features
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns to match model
        input_encoded = input_encoded[model_features]

        # Make prediction
        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
