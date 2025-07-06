import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, encoders, and original dataset
model = joblib.load("models/churn_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
data = pd.read_csv("customer_churn.csv")

st.title("Customer Churn Prediction")

# --- Input Customer ID ---
customerID = st.text_input("Enter Customer ID")

# Check if customerID exists
existing_customer = data[data["customerID"] == customerID]

# If exists, prefill fields
if not existing_customer.empty:
    st.success("Existing customer found. Prefilling data.")
    input_data = existing_customer.iloc[0]
else:
    st.warning("New customer. Please fill out the details.")
    input_data = {}

# --- Collect inputs (prefilled if existing) ---
def get_value(col, options=None, is_numeric=False, default=None):
    if isinstance(input_data, pd.Series) and col in input_data:
        val = input_data[col]
    else:
        val = default

    if options:
        return st.selectbox(col, options, index=options.index(val) if val in options else 0)
    elif is_numeric:
        return st.number_input(col, value=float(val) if val else 0.0)
    else:
        return st.text_input(col, value=val if val else "")

gender = get_value("gender", ["Male", "Female"])
SeniorCitizen = get_value("SeniorCitizen", [0, 1])
Partner = get_value("Partner", ["Yes", "No"])
#Dependents = get_value("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, int(input_data["tenure"]) if "tenure" in input_data else 12)
#PhoneService = get_value("PhoneService", ["Yes", "No"])
MultipleLines = get_value("MultipleLines", ["Yes", "No", "No phone service"])
InternetService = get_value("InternetService", ["DSL", "Fiber optic", "No"])
OnlineSecurity = get_value("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = get_value("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = get_value("DeviceProtection", ["Yes", "No", "No internet service"])
TechSupport = get_value("TechSupport", ["Yes", "No", "No internet service"])
#StreamingTV = get_value("StreamingTV", ["Yes", "No", "No internet service"])
#StreamingMovies = get_value("StreamingMovies", ["Yes", "No", "No internet service"])
Contract = get_value("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = get_value("PaperlessBilling", ["Yes", "No"])
PaymentMethod = get_value("PaymentMethod", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = get_value("MonthlyCharges", is_numeric=True, default=50.0)
TotalCharges = get_value("TotalCharges", is_numeric=True, default=100.0)

# --- Predict ---
if st.button("Predict"):
    input_dict = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'tenure': tenure,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            input_df[col] = le.transform(input_df[col].astype(str))


    # Predict class and probability
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred] * 100  # Get probability of the predicted class
    # Display result
    st.subheader("Result")
    if pred == 1:
        st.success(f"Prediction: Churn ({proba:.2f}%)")
    else:
        st.success(f"Prediction: No Churn ({proba:.2f}%)")
