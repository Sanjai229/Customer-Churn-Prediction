from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    values = np.array(list(data.values())).reshape(1, -1)
    scaled = scaler.transform(values)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
