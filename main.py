import numpy as np
import pandas as pd
from fastapi import FastAPI
import joblib

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define all features in the exact same order as training
model_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
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
    'PaymentMethod_Mailed check'
]

@app.post("/predict")
def predict(data: dict):
    try:
        # Create a DataFrame with all features
        input_df = pd.DataFrame(columns=model_features)
        input_df.loc[0] = 0  # default all zeros

        # Fill numeric features
        for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in data:
                input_df[col] = data[col]

        # Fill categorical features with one-hot encoding
        for col in data:
            if col in model_features and col not in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
                input_df[col] = 1

        # Scale numeric columns
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
