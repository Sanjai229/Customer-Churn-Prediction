from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")  # make sure this matches your model filename
scaler = joblib.load("scaler.pkl")        # optional, only if you have a scaler

# Home endpoint (GET)
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

# Health check endpoint (HEAD)
@app.head("/")
def health_check():
    return PlainTextResponse("OK")

# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        # Make sure all features expected by the model are present
        expected_features = [
            "tenure", "MonthlyCharges", "TotalCharges", 
            "gender", "SeniorCitizen", "Partner", "Dependents",
            # add all other features here exactly as in training
        ]
        
        # Fill missing features with 0 (or a default value)
        input_values = [data.get(f, 0) for f in expected_features]

        # Convert to numpy array and scale
        values = np.array(input_values).reshape(1, -1)
        scaled = scaler.transform(values) if scaler else values

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}

