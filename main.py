from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names used by the model
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

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

class InputData(BaseModel):
    SeniorCitizen: int = 0
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str = "Male"
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "No"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"

@app.get("/")
def health_check():
    return {"status": "API is live"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to DataFrame
        input_dict = data.dict()

        # Handle categorical one-hot encoding manually
        df = pd.DataFrame([0]*len(model_features), index=model_features).T

        # Numeric columns: scale
        for col in numeric_cols:
            df[col] = scaler.transform([[input_dict[col]]])[0]

        # One-hot encode categorical variables
        if input_dict['gender'] == "Male":
            df['gender_Male'] = 1
        if input_dict['Partner'] == "Yes":
            df['Partner_Yes'] = 1
        if input_dict['Dependents'] == "Yes":
            df['Dependents_Yes'] = 1
        if input_dict['PhoneService'] == "Yes":
            df['PhoneService_Yes'] = 1
        if input_dict['MultipleLines'] == "Yes":
            df['MultipleLines_Yes'] = 1
        if input_dict['MultipleLines'] == "No phone service":
            df['MultipleLines_No phone service'] = 1
        if input_dict['InternetService'] == "Fiber optic":
            df['InternetService_Fiber optic'] = 1
        if input_dict['InternetService'] == "No":
            df['InternetService_No'] = 1
        if input_dict['OnlineSecurity'] == "Yes":
            df['OnlineSecurity_Yes'] = 1
        if input_dict['OnlineSecurity'] == "No internet service":
            df['OnlineSecurity_No internet service'] = 1
        if input_dict['OnlineBackup'] == "Yes":
            df['OnlineBackup_Yes'] = 1
        if input_dict['OnlineBackup'] == "No internet service":
            df['OnlineBackup_No internet service'] = 1
        if input_dict['DeviceProtection'] == "Yes":
            df['DeviceProtection_Yes'] = 1
        if input_dict['DeviceProtection'] == "No internet service":
            df['DeviceProtection_No internet service'] = 1
        if input_dict['TechSupport'] == "Yes":
            df['TechSupport_Yes'] = 1
        if input_dict['TechSupport'] == "No internet service":
            df['TechSupport_No internet service'] = 1
        if input_dict['StreamingTV'] == "Yes":
            df['StreamingTV_Yes'] = 1
        if input_dict['StreamingTV'] == "No internet service":
            df['StreamingTV_No internet service'] = 1
        if input_dict['StreamingMovies'] == "Yes":
            df['StreamingMovies_Yes'] = 1
        if input_dict['StreamingMovies'] == "No internet service":
            df['StreamingMovies_No internet service'] = 1
        if input_dict['Contract'] == "One year":
            df['Contract_One year'] = 1
        if input_dict['Contract'] == "Two year":
            df['Contract_Two year'] = 1
        if input_dict['PaperlessBilling'] == "Yes":
            df['PaperlessBilling_Yes'] = 1
        if input_dict['PaymentMethod'] == "Credit card (automatic)":
            df['PaymentMethod_Credit card (automatic)'] = 1
        if input_dict['PaymentMethod'] == "Electronic check":
            df['PaymentMethod_Electronic check'] = 1
        if input_dict['PaymentMethod'] == "Mailed check":
            df['PaymentMethod_Mailed check'] = 1

        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
