
import joblib
import pandas as pd
from pathlib import Path
from src.feature_encoder import encode_customer
# Load the trained model and feature names
model = joblib.load("models/logistic_regression_model.joblib")

data = pd.read_csv("data/processed/encoded_churn.csv")
feature_names = data.drop(columns=["Churn"]).columns


def ask_choice(prompt, choices):
    """Ask until the user enters one of the allowed choices."""
    choices_dict = {choice.lower(): choice for choice in choices}

    while True:
        answer = input(prompt).strip().lower()
        if answer in choices_dict:
            return choices_dict[answer]

        print("Invalid choice. Please choose:", ", ".join(choices))


print("Customer Churn Prediction")
print("-------------------------")
print("Enter the customer's details.\n")

gender = ask_choice("Gender (Male/Female): ", ["Male", "Female"])
senior_citizen = int(
    ask_choice("Senior citizen? (1 = Yes, 0 = No): ", ["0", "1"])
)
partner = ask_choice("Has a partner? (Yes/No): ", ["Yes", "No"])
dependents = ask_choice("Has dependents? (Yes/No): ", ["Yes", "No"])

tenure = int(input("Tenure in months (0-72): "))
phone_service = ask_choice(
    "Has phone service? (Yes/No): ", ["Yes", "No"]
)
multiple_lines = ask_choice(
    "Multiple lines (Yes/No/No phone service): ",
    ["Yes", "No", "No phone service"]
)

internet_service = ask_choice(
    "Internet service (DSL/Fiber optic/No): ",
    ["DSL", "Fiber optic", "No"]
)

online_security = ask_choice(
    "Online security (Yes/No/No internet service): ",
    ["Yes", "No", "No internet service"]
)
online_backup = ask_choice(
    "Online backup (Yes/No/No internet service): ",
    ["Yes", "No", "No internet service"]
)
device_protection = ask_choice(
    "Device protection (Yes/No/No internet service): ",
    ["Yes", "No", "No internet service"]
)
tech_support = ask_choice(
    "Tech support (Yes/No/No internet service): ",
    ["Yes", "No", "No internet service"]
)
streaming_tv = ask_choice(
    "Streaming TV (Yes/No/No internet service): ",
    ["Yes", "No", "No internet service"]
)
streaming_movies = ask_choice(
    "Streaming movies (Yes/No/No internet service): ",
    ["Yes", "No", "No internet service"]
)

contract = ask_choice(
    "Contract (Month-to-month/One year/Two year): ",
    ["Month-to-month", "One year", "Two year"]
)
paperless_billing = ask_choice(
    "Paperless billing? (Yes/No): ", ["Yes", "No"]
)
payment_method = ask_choice(
    "Payment method (Electronic check/Mailed check/"
    "Bank transfer (automatic)/Credit card (automatic)): ",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = float(input("Monthly charges: "))
total_charges = float(input("Total charges: "))

# Store customer answers in the format expected by the encoder
customer = {
    "senior_citizen": "Yes" if senior_citizen == 1 else "No",
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

# Encode customer data using the reusable encoder
customer_df = encode_customer(customer, feature_names)
# Predict churn
prediction = model.predict(customer_df)[0]
churn_probability = model.predict_proba(customer_df)[0][1]
print("\nPrediction Result")
print("-----------------")
print(
    "Prediction:",
    "Likely to churn" if prediction == 1 else "Unlikely to churn"
)
print(f"Estimated churn probability: {churn_probability:.2%}")
