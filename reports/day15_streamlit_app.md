\# Day 15: Streamlit Customer Churn Prediction App



\## Objective

Build a web interface for the Customer Churn Prediction project and connect it to the trained Logistic Regression model.



\## Implementation

\- Created `app.py` using Streamlit.

\- Added customer input fields for tenure, monthly charges, total charges, contract type, internet service, and payment method.

\- Loaded the trained model from `models/logistic\_regression\_model.joblib`.

\- Loaded the encoded dataset to retrieve the model's expected feature names and order.

\- Used `predict()` to generate a churn classification.

\- Used `predict\_proba()` to calculate the estimated churn probability.

\- Displayed the prediction, probability, and progress bar in the browser.



\## Test Results

Tested the application through the Streamlit web interface.



\- Prediction: Unlikely to churn

\- Estimated churn probability: 12.85%

\- Application status: Prediction displayed successfully.



\## Key Learnings

\- Building interactive web interfaces with Streamlit.

\- Loading a saved machine learning model using Joblib.

\- Converting user inputs into model features.

\- Displaying classification results and probabilities.



\## Limitations

\- The current form collects only six customer attributes.

\- The remaining model features are initialized to zero, which can affect prediction quality.

\- The probability is a model estimate, not a guarantee of customer behavior.



\## Next Steps

\- Collect all required customer attributes.

\- Improve input validation and prediction reliability.

\- Enhance the interface and test different customer profiles.



