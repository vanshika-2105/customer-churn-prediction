
import pandas as pd


def encode_customer(customer, feature_names):
    """
    Convert customer information into the 30 features
    expected by the trained churn prediction model.
    """

    # Start with all features set to zero
    features = {feature: 0 for feature in feature_names}

    # Numerical features
    features["SeniorCitizen"] = int(customer["senior_citizen"] == "Yes")
    features["tenure"] = customer["tenure"]
    features["MonthlyCharges"] = customer["monthly_charges"]
    features["TotalCharges"] = customer["total_charges"]

    # Personal features
    features["gender_Male"] = int(customer["gender"] == "Male")
    features["Partner_Yes"] = int(customer["partner"] == "Yes")
    features["Dependents_Yes"] = int(customer["dependents"] == "Yes")

    # Phone features
    features["PhoneService_Yes"] = int(customer["phone_service"] == "Yes")

    features["MultipleLines_No phone service"] = int(
        customer["multiple_lines"] == "No phone service"
    )

    features["MultipleLines_Yes"] = int(
        customer["multiple_lines"] == "Yes"
    )

    # Internet features
    features["InternetService_Fiber optic"] = int(
        customer["internet_service"] == "Fiber optic"
    )

    features["InternetService_No"] = int(
        customer["internet_service"] == "No"
    )

    # Other internet services
    service_names = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    key = {
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streamingtv",
        "StreamingMovies": "streamingmovies"
    }

    for name in service_names:
        value = customer[key[name]]

        features[f"{name}_No internet service"] = int(
            value == "No internet service"
        )

        features[f"{name}_Yes"] = int(value == "Yes")

    # Account features
    features["Contract_One year"] = int(
        customer["contract"] == "One year"
    )

    features["Contract_Two year"] = int(
        customer["contract"] == "Two year"
    )

    features["PaperlessBilling_Yes"] = int(
        customer["paperless_billing"] == "Yes"
    )

    features["PaymentMethod_Credit card (automatic)"] = int(
        customer["payment_method"] == "Credit card (automatic)"
    )

    features["PaymentMethod_Electronic check"] = int(
        customer["payment_method"] == "Electronic check"
    )

    features["PaymentMethod_Mailed check"] = int(
        customer["payment_method"] == "Mailed check"
    )

    # Arrange features in the exact training order
    customer_df = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    return customer_df
