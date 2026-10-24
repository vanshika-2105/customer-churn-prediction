# Day 11 — Feature Importance Analysis

## 1. Objective

Identify the customer characteristics most strongly associated with churn predictions made by the Logistic Regression model.

## 2. Experimental Setup

- Dataset: Telco Customer Churn
- Model: Logistic Regression
- Preprocessing: StandardScaler and encoded categorical features
- Train-test split: 80% / 20%
- Random state: 42
- Analysis method: Logistic Regression coefficient magnitude

## 3. Top 10 Features

| Feature | Coefficient |
|---|---:|
| tenure | -1.2365 |
| MonthlyCharges | -0.9202 |
| InternetService_Fiber optic | 0.7762 |
| Contract_Two year | -0.5869 |
| TotalCharges | 0.5143 |
| Contract_One year | -0.2855 |
| StreamingMovies_Yes | 0.2572 |
| StreamingTV_Yes | 0.2571 |
| MultipleLines_Yes | 0.2162 |
| PaperlessBilling_Yes | 0.1820 |

## 4. Interpretation

### Tenure

Tenure has the largest absolute coefficient in this analysis. Its negative coefficient indicates that higher tenure is associated with lower predicted churn, holding other encoded features constant.

### Monthly Charges

MonthlyCharges has a negative coefficient in this fitted model. This is a conditional model association and should not be interpreted as evidence that higher monthly charges generally prevent churn.

### Internet Service

The Fiber optic indicator has a positive coefficient, meaning it pushes predictions toward churn when other encoded features are held constant.

### Contract Type

The one-year and two-year contract indicators have negative coefficients relative to the reference contract category. Contract type may therefore be useful when identifying customers at risk.

### Total Charges and Other Features

TotalCharges, streaming service indicators, multiple lines, and paperless billing also contribute to the model's predictions.

## 5. Important Limitations

- Coefficients show associations, not causation.
- Categorical coefficients are interpreted relative to their reference categories.
- Correlated features can affect coefficient magnitudes and signs.
- Coefficient magnitude is not a complete measure of predictive importance.
- These findings describe the current fitted model and dataset.

## 6. Conclusion

Tenure, monthly charges, fiber-optic internet service, and contract type have comparatively large absolute coefficients in the current Logistic Regression model.

These findings can guide further analysis of customer churn patterns, but they should be combined with model evaluation and business context before making retention decisions.

## 7. Next Steps

- Visualize the most important coefficients.
- Compare feature importance with a Decision Tree model.
- Investigate customer groups with higher predicted churn risk.