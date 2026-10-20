# Customer Churn Prediction — Model Evaluation

## 1. Model Overview

- **Model:** Logistic Regression
- **Dataset:** IBM Telco Customer Churn
- **Total customer records:** 7,043
- **Training records:** 5,634
- **Testing records:** 1,409
- **Input features:** 30

## 2. Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 80.70% |
| Precision | 65.84% |
| Recall | 56.68% |
| F1-score | 60.92% |

## 3. Confusion Matrix

| Actual / Predicted | Stayed (0) | Left (1) |
|---|---:|---:|
| Stayed (0) | 925 | 110 |
| Left (1) | 162 | 212 |

## 4. Findings

- The model correctly classified approximately 80.7% of test customers.
- Of customers predicted to leave, approximately 65.84% actually left.
- The model identified approximately 56.68% of customers who actually left.
- The model missed 162 customers who churned and incorrectly flagged 110 customers who stayed.

## 5. Conclusion

Logistic Regression provides a useful baseline for customer churn prediction. Its accuracy is approximately 80.7%, but its recall of 56.68% indicates that it misses a substantial number of customers who leave.

Further work should explore improving recall, comparing other classification algorithms, and tuning the decision threshold. The model's performance should be evaluated carefully before using it for real business decisions.