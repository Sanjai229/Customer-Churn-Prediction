from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load your model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")  # fitted on numeric columns only

# Numeric and categorical columns
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Exact feature order your model expects
MODEL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(float)

    # Scale numeric columns
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Add missing features
    for col in MODEL_FEATURES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns
    df_encoded = df_encoded[MODEL_FEATURES]

    return df_encoded.values

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        input_array = preprocess_input(data)
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]
        return {"prediction": int(pred), "probability": float(prob)}
    except Exception as e:
        return {"error": str(e)}
