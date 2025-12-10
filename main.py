from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")  # fitted on numeric columns only

# Feature order expected by the model (after preprocessing)
MODEL_FEATURES = [
    'gender_Female', 'gender_Male', 'SeniorCitizen', 'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes', 'tenure', 'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_Yes', 'OnlineBackup_No',
    'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes', 'TechSupport_No',
    'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_Yes', 'StreamingMovies_No',
    'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer',
    'PaymentMethod_Credit card', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'MonthlyCharges', 'TotalCharges'
]

# Preprocess function
def preprocess_input(data: dict):
    # Convert categorical features to one-hot manually
    df = pd.DataFrame([data])

    # Scale numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # One-hot encode categorical columns (using same names as MODEL_FEATURES)
    df_encoded = pd.get_dummies(df)

    # Ensure all expected features exist
    for col in MODEL_FEATURES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match model
    df_encoded = df_encoded[MODEL_FEATURES]

    return df_encoded.values

# Health check
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

# Predict endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        input_array = preprocess_input(data)
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]
        return {"prediction": int(pred), "probability": float(prob)}
    except Exception as e:
        return {"error": str(e)}
