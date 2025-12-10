from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# List of features your model expects
MODEL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 
                  'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
                  'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 
                  'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
                  'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 
                  'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 
                  'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
                  'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input dict to DataFrame
        df = pd.DataFrame([data])

        # Scale numeric columns
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # One-hot encode categorical columns
        df = pd.get_dummies(df)

        # Add missing columns with 0
        for col in MODEL_FEATURES:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns
        df = df[MODEL_FEATURES]

        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
