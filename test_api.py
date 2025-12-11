import requests

url = "https://customer-churn-prediction-2r5b.onrender.com/predict"

data = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 75.5,
    "TotalCharges": 910.5,
    "gender_Male": 1,  # one-hot encoding
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    # Add any other categorical features if needed, or leave them 0
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response:", response.json())
