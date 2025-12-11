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

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# JSON prediction endpoint (for API)
@app.post("/predict")
def predict(data: dict):
    try:
        input_df = pd.DataFrame(columns=model_features)
        input_df.loc[0] = 0

        # Numeric features
        for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in data:
                input_df[col] = data[col]

        # Categorical features
        for col in data:
            if col in model_features and col not in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
                input_df[col] = 1

        # Scale numeric columns
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return {"prediction": int(pred), "probability": float(prob)}

    except Exception as e:
        return {"error": str(e)}


# Form submission endpoint (for HTML)
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    SeniorCitizen: int = Form(...),
    tenure: float = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    gender: str = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    PaperlessBilling: str = Form(...)
):
    try:
        input_df = pd.DataFrame(columns=model_features)
        input_df.loc[0] = 0

        # Numeric features
        input_df['SeniorCitizen'] = SeniorCitizen
        input_df['tenure'] = tenure
        input_df['MonthlyCharges'] = MonthlyCharges
        input_df['TotalCharges'] = TotalCharges

        # Categorical features mapping to model columns
        if gender == "Male":
            input_df['gender_Male'] = 1
        if Partner == "Yes":
            input_df['Partner_Yes'] = 1
        if Dependents == "Yes":
            input_df['Dependents_Yes'] = 1
        if PaperlessBilling == "Yes":
            input_df['PaperlessBilling_Yes'] = 1

        # Scale numeric columns
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": int(pred),
            "probability": round(float(prob), 2)
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })
