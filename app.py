
import streamlit as st
import pandas as pd
import joblib

# Configure the web page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load model and expected feature names
model = joblib.load("models/logistic_regression_model.joblib")
encoded_data = pd.read_csv("data/processed/encoded_churn.csv")
feature_names = encoded_data.drop(columns=["Churn"]).columns.tolist()

st.title("Customer Churn Prediction")
st.write("Enter the customer's details to estimate churn risk.")

with st.form("customer_form"):
    st.subheader("Personal Information")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    st.subheader("Account Information")

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )
    contract = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing?", ["No", "Yes"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=85.50
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=1026.0
    )

    st.subheader("Services")

  
    phone_service = st.selectbox("Phone Service?", ["No", "Yes"])

    if phone_service == "No":
        multiple_lines_options = ["No phone service"]
    else:
        multiple_lines_options = ["No", "Yes"]

    multiple_lines = st.selectbox(
        "Multiple Lines", multiple_lines_options
    )

  
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    if internet_service == "No":
        internet_service_options = ["No internet service"]
    else:
        internet_service_options = ["No", "Yes"]

    online_security = st.selectbox(
        "Online Security", internet_service_options
    )
    online_backup = st.selectbox(
        "Online Backup", internet_service_options
    )
    device_protection = st.selectbox(
        "Device Protection", internet_service_options
    )
    tech_support = st.selectbox(
        "Tech Support", internet_service_options
    )
    streaming_tv = st.selectbox(
        "Streaming TV", internet_service_options
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", internet_service_options
    )


    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Start with all expected model features set to zero
    customer = {feature: 0 for feature in feature_names}

    # Numerical features
    customer["SeniorCitizen"] = int(senior_citizen == "Yes")
    customer["tenure"] = tenure
    customer["MonthlyCharges"] = monthly_charges
    customer["TotalCharges"] = total_charges

    # Personal features
    customer["gender_Male"] = int(gender == "Male")
    customer["Partner_Yes"] = int(partner == "Yes")
    customer["Dependents_Yes"] = int(dependents == "Yes")

    # Phone features
    customer["PhoneService_Yes"] = int(phone_service == "Yes")
    customer["MultipleLines_No phone service"] = int(
        multiple_lines == "No phone service"
    )
    customer["MultipleLines_Yes"] = int(multiple_lines == "Yes")

    # Internet features
    customer["InternetService_Fiber optic"] = int(
        internet_service == "Fiber optic"
    )
    customer["InternetService_No"] = int(internet_service == "No")

    # Other service features
    service_values = {
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies
    }

    for feature, value in service_values.items():
        customer[f"{feature}_No internet service"] = int(
            value == "No internet service"
        )
        customer[f"{feature}_Yes"] = int(value == "Yes")

    # Account features
    customer["Contract_One year"] = int(contract == "One year")
    customer["Contract_Two year"] = int(contract == "Two year")
    customer["PaperlessBilling_Yes"] = int(
        paperless_billing == "Yes"
    )
    customer["PaymentMethod_Credit card (automatic)"] = int(
        payment_method == "Credit card (automatic)"
    )
    customer["PaymentMethod_Electronic check"] = int(
        payment_method == "Electronic check"
    )
    customer["PaymentMethod_Mailed check"] = int(
        payment_method == "Mailed check"
    )

    # Arrange features in the exact training order
    customer_df = pd.DataFrame(
        [[customer[feature] for feature in feature_names]],
        columns=feature_names
    )

    # Generate prediction
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Likely to churn")
    else:
        st.success("Prediction: Unlikely to churn")

    st.metric("Estimated churn probability", f"{probability:.2%}")
    st.progress(float(probability))

    st.caption(
        "This result is a model estimate, not a guarantee of customer behavior."
    )
