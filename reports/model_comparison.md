# Customer Churn Prediction — Model Comparison

## 1. Objective

Compare Logistic Regression and Decision Tree classifiers using the same training and testing split.

## 2. Results

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 80.70% | 79.42% |
| Precision | 65.84% | 63.12% |
| Recall | 56.68% | 54.01% |
| F1-score | 60.92% | 58.21% |

## 3. Findings

- Logistic Regression achieved higher accuracy, precision, recall, and F1-score.
- Logistic Regression identified more customers who actually churned.
- The Decision Tree used a maximum depth of five to help control overfitting.
- The current results do not establish that Logistic Regression will perform better on every dataset or under every configuration.

## 4. Conclusion

Logistic Regression is the stronger baseline among these two configurations on the current test set. Further work could tune the Decision Tree, adjust classification thresholds, and evaluate additional models. Model selection should consider churn recall alongside precision and other business requirements.