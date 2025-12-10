import requests

url = "https://customer-churn-prediction-2r5b.onrender.com/predict"

# Sample input
data = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 75.5,
    "TotalCharges": 910.5,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response:", response.json())
