import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load saved model and scaler
# -------------------------------


try:
    model = joblib.load("rf_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

st.title("📊 Telco Customer Churn Prediction App")
st.write("Enter customer details to predict whether they will churn.")

# -------------------------------
# Input fields
# -------------------------------
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0)

# -------------------------------
# Convert inputs to DataFrame
# -------------------------------
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding
input_df = pd.get_dummies(input_df)

# -------------------------------
# Align columns with training data
# -------------------------------
try:
    train_cols = model.feature_names_in_
    
    # Ensure numeric columns exist
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reindex to match model columns
    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    # Scale numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
except Exception as e:
    st.error(f"Error preparing input: {e}")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer WILL CHURN (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------
# Debug info (optional)
# -------------------------------
if st.checkbox("Show Debug Info"):
    st.write("Input DataFrame columns:", list(input_df.columns))
    st.write("Model expected columns:", list(train_cols))
