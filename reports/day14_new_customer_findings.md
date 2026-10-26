
# Day 14: Predict Churn for a New Customer

## Objective
Extend the customer churn project to accept manually entered customer details and predict churn using the trained Logistic Regression model.

## Implementation
- Created `src/predict_new_customer.py`.
- Loaded the saved model from `models/logistic_regression_model.joblib`.
- Collected customer information through terminal inputs.
- Validated categorical answers using the `ask_choice()` function.
- Converted customer details into the 30 encoded features expected by the model.
- Generated a churn prediction and estimated churn probability.

## Test Results

Tested the script using manually entered sample customer details.

- Tenure: 12 months
- Monthly charges: 85.50
- Total charges: 1026.00
- Contract: Month-to-month
- Internet service: Fiber optic
- Payment method: Electronic check

**Prediction:** Likely to churn

**Estimated churn probability:** 74.89%

## Key Learnings
- How to collect user input in Python.
- How to convert categorical values into numerical features.
- Why model input feature names and order must match training.
- How to use `predict()` and `predict_proba()` for classification.

## Limitations
- Input validation for numeric values and ranges can be improved.
- The prediction is an estimate, not a guarantee of customer behavior.
- The current workflow runs in the terminal rather than a graphical application.

## Output Files
- `src/predict_new_customer.py`
- `reports/day14_new_customer_findings.md`
