from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")  # Only numeric columns are scaled

# Define model features
model_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
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
    'PaymentMethod_Mailed check'
]

numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Start with zeros for all features
        input_df = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

        # Fill numeric columns
        for col in numeric_cols:
            if col in data:
                input_df[col] = data[col]

        # Fill categorical columns using one-hot
        for col in data:
            if col not in numeric_cols:
                feature_name = f"{col}_{data[col]}"
                if feature_name in input_df.columns:
                    input_df[feature_name] = 1

        # Scale numeric columns
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
