
import streamlit as st
import pandas as pd
import joblib

# Configure the page
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# Load the trained model and feature names
model = joblib.load("models/logistic_regression_model.joblib")

encoded_data = pd.read_csv("data/processed/encoded_churn.csv")
feature_names = encoded_data.drop(columns=["Churn"]).columns.tolist()

# Page heading
st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate their churn risk.")

# Customer form
st.subheader("Customer Information")

with st.form("customer_form"):

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=85.50
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=1026.0
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
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

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create the model input with all 30 expected features
    customer = {feature: 0 for feature in feature_names}

    customer["tenure"] = tenure
    customer["MonthlyCharges"] = monthly_charges
    customer["TotalCharges"] = total_charges

    # Encode contract type
    customer["Contract_One year"] = int(contract == "One year")
    customer["Contract_Two year"] = int(contract == "Two year")

    # Encode internet service
    customer["InternetService_Fiber optic"] = int(
        internet_service == "Fiber optic"
    )
    customer["InternetService_No"] = int(internet_service == "No")

    # Encode payment method
    customer["PaymentMethod_Credit card (automatic)"] = int(
        payment_method == "Credit card (automatic)"
    )
    customer["PaymentMethod_Electronic check"] = int(
        payment_method == "Electronic check"
    )
    customer["PaymentMethod_Mailed check"] = int(
        payment_method == "Mailed check"
    )

    # Create DataFrame in the exact training feature order
    customer_df = pd.DataFrame(
        [[customer[feature] for feature in feature_names]],
        columns=feature_names
    )

    # Make prediction
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Likely to churn")
    else:
        st.success("Prediction: Unlikely to churn")

    st.metric(
        "Estimated churn probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.caption(
        "This is a model estimate, not a guarantee of customer behavior. "
        "Some customer fields are not collected in this first version."
    )
