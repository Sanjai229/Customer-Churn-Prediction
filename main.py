from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

# Load your trained model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service',
    'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service',
    'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

@app.get("/")
def health_check():
    return {"status": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Initialize input array
        input_array = np.zeros((1, len(numeric_cols) + len(categorical_cols)))

        # Fill numeric features
        for i, col in enumerate(numeric_cols):
            input_array[0, i] = float(data.get(col, 0))

        # Mapping for categorical inputs
        mapping = {
            "gender": "gender_Male",
            "Partner": "Partner_Yes",
            "Dependents": "Dependents_Yes",
            "PhoneService": "PhoneService_Yes",
            "MultipleLines": {"Yes": "MultipleLines_Yes", "No phone service": "MultipleLines_No phone service"},
            "InternetService": {"Fiber optic": "InternetService_Fiber optic", "No": "InternetService_No"},
            "OnlineSecurity": {"Yes": "OnlineSecurity_Yes", "No internet service": "OnlineSecurity_No internet service"},
            "OnlineBackup": {"Yes": "OnlineBackup_Yes", "No internet service": "OnlineBackup_No internet service"},
            "DeviceProtection": {"Yes": "DeviceProtection_Yes", "No internet service": "DeviceProtection_No internet service"},
            "TechSupport": {"Yes": "TechSupport_Yes", "No internet service": "TechSupport_No internet service"},
            "StreamingTV": {"Yes": "StreamingTV_Yes", "No internet service": "StreamingTV_No internet service"},
            "StreamingMovies": {"Yes": "StreamingMovies_Yes", "No internet service": "StreamingMovies_No internet service"},
            "Contract": {"One year": "Contract_One year", "Two year": "Contract_Two year"},
            "PaperlessBilling": {"Yes": "PaperlessBilling_Yes"},
            "PaymentMethod": {"Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
                              "Electronic check": "PaymentMethod_Electronic check",
                              "Mailed check": "PaymentMethod_Mailed check"}
        }

        # Apply one-hot mapping
        for key, val in mapping.items():
            if key in data:
                if isinstance(val, dict):
                    one_hot_col = val.get(data[key])
                else:
                    one_hot_col = val
                if one_hot_col and one_hot_col in categorical_cols:
                    idx = len(numeric_cols) + categorical_cols.index(one_hot_col)
                    input_array[0, idx] = 1

        # Scale numeric features
        input_array[:, :len(numeric_cols)] = scaler.transform(input_array[:, :len(numeric_cols)])

        # Make prediction
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}
