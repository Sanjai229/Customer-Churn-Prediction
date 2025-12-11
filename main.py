from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and scaler
model = joblib.load("rf_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Features in exact order
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

# Root endpoint serves HTML form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Form submission endpoint
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()
    data = dict(form)

    # Convert numeric fields
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        data[col] = float(data[col])

    # Create input DataFrame
    input_df = pd.DataFrame(columns=model_features)
    input_df.loc[0] = 0

    # Fill numeric
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        input_df[col] = data[col]

    # Fill categorical
    if data.get('gender') == 'Male':
        input_df['gender_Male'] = 1
    if data.get('Partner') == 'Yes':
        input_df['Partner_Yes'] = 1
    if data.get('Dependents') == 'Yes':
        input_df['Dependents_Yes'] = 1
    if data.get('PaperlessBilling') == 'Yes':
        input_df['PaperlessBilling_Yes'] = 1

    # Scale numeric
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": int(pred), "probability": float(prob)}
    )
