
import pandas as pd

from src.feature_encoder import encode_customer


def test_encoder_creates_30_features():
    feature_names = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "gender_Male", "Partner_Yes", "Dependents_Yes",
        "PhoneService_Yes", "MultipleLines_No phone service",
        "MultipleLines_Yes", "InternetService_Fiber optic",
        "InternetService_No",
        "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
        "OnlineBackup_No internet service", "OnlineBackup_Yes",
        "DeviceProtection_No internet service", "DeviceProtection_Yes",
        "TechSupport_No internet service", "TechSupport_Yes",
        "StreamingTV_No internet service", "StreamingTV_Yes",
        "StreamingMovies_No internet service", "StreamingMovies_Yes",
        "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
    ]

    customer = {
        "senior_citizen": "No",
        "tenure": 12,
        "monthly_charges": 85.50,
        "total_charges": 1026.0,
        "gender": "Female",
        "partner": "Yes",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No",
        "streamingtv": "Yes",
        "streamingmovies": "No",
        "contract": "One year",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check"
    }

    result = encode_customer(customer, feature_names)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 30)
    assert list(result.columns) == feature_names
def test_encoder_converts_customer_values_correctly():
    feature_names = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "gender_Male", "Partner_Yes", "Dependents_Yes",
        "PhoneService_Yes", "MultipleLines_No phone service",
        "MultipleLines_Yes", "InternetService_Fiber optic",
        "InternetService_No",
        "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
        "OnlineBackup_No internet service", "OnlineBackup_Yes",
        "DeviceProtection_No internet service", "DeviceProtection_Yes",
        "TechSupport_No internet service", "TechSupport_Yes",
        "StreamingTV_No internet service", "StreamingTV_Yes",
        "StreamingMovies_No internet service", "StreamingMovies_Yes",
        "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
    ]

    customer = {
        "senior_citizen": "Yes",
        "tenure": 24,
        "monthly_charges": 90.0,
        "total_charges": 2160.0,
        "gender": "Male",
        "partner": "Yes",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "Yes",
        "tech_support": "No",
        "streamingtv": "Yes",
        "streamingmovies": "No",
        "contract": "One year",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check"
    }

    result = encode_customer(customer, feature_names)

    assert result.loc[0, "SeniorCitizen"] == 1
    assert result.loc[0, "tenure"] == 24
    assert result.loc[0, "gender_Male"] == 1
    assert result.loc[0, "Partner_Yes"] == 1
    assert result.loc[0, "InternetService_Fiber optic"] == 1
    assert result.loc[0, "OnlineSecurity_Yes"] == 1
    assert result.loc[0, "Contract_One year"] == 1
    assert result.loc[0, "PaymentMethod_Electronic check"] == 1

    result = encode_customer(customer, feature_names)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 30)
    assert list(result.columns) == feature_names
