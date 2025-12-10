import requests

url = "https://customer-churn-prediction-2r5b.onrender.com/predict"

# Example input (19 features, match your model training)
data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 910.5
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response text:", response.text)

if response.status_code == 200:
    print("JSON response:", response.json())
else:
    print("Failed to get valid response")
