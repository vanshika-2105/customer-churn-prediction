
import joblib
import pandas as pd
from pathlib import Path

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

# Convert the answers into the same features used during training
customer = {
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender_Male": int(gender == "Male"),
    "Partner_Yes": int(partner == "Yes"),
    "Dependents_Yes": int(dependents == "Yes"),
    "PhoneService_Yes": int(phone_service == "Yes"),
    "MultipleLines_No phone service": int(
        multiple_lines == "No phone service"
    ),
    "MultipleLines_Yes": int(multiple_lines == "Yes"),
    "InternetService_Fiber optic": int(
        internet_service == "Fiber optic"
    ),
    "InternetService_No": int(internet_service == "No"),
    "OnlineSecurity_No internet service": int(
        online_security == "No internet service"
    ),
    "OnlineSecurity_Yes": int(online_security == "Yes"),
    "OnlineBackup_No internet service": int(
        online_backup == "No internet service"
    ),
    "OnlineBackup_Yes": int(online_backup == "Yes"),
    "DeviceProtection_No internet service": int(
        device_protection == "No internet service"
    ),
    "DeviceProtection_Yes": int(device_protection == "Yes"),
    "TechSupport_No internet service": int(
        tech_support == "No internet service"
    ),
    "TechSupport_Yes": int(tech_support == "Yes"),
    "StreamingTV_No internet service": int(
        streaming_tv == "No internet service"
    ),
    "StreamingTV_Yes": int(streaming_tv == "Yes"),
    "StreamingMovies_No internet service": int(
        streaming_movies == "No internet service"
    ),
    "StreamingMovies_Yes": int(streaming_movies == "Yes"),
    "Contract_One year": int(contract == "One year"),
    "Contract_Two year": int(contract == "Two year"),
    "PaperlessBilling_Yes": int(paperless_billing == "Yes"),
    "PaymentMethod_Credit card (automatic)": int(
        payment_method == "Credit card (automatic)"
    ),
    "PaymentMethod_Electronic check": int(
        payment_method == "Electronic check"
    ),
    "PaymentMethod_Mailed check": int(
        payment_method == "Mailed check"
    )
}

# Arrange the features in the exact training order
customer_df = pd.DataFrame([customer]).reindex(columns=feature_names)

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
