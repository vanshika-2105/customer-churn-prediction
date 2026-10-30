
import streamlit as st
import pandas as pd
import joblib
from src.feature_encoder import encode_customer
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

    # Collect customer information
    customer = {
        "senior_citizen": senior_citizen,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,

        "gender": gender,
        "partner": partner,
        "dependents": dependents,

        "phone_service": phone_service,
        "multiple_lines": multiple_lines,

        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streamingtv": streaming_tv,
        "streamingmovies": streaming_movies,

        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method
    }

    # Convert customer details into model features
    customer_df = encode_customer(customer, feature_names)

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
