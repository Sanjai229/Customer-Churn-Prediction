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
    # Convert input data to numpy array
    values = np.array(list(data.values())).reshape(1, -1)

    # Scale if scaler exists
    scaled = scaler.transform(values) if scaler else values

    # Make prediction
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
