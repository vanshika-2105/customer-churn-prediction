# Day 9 — Classification Threshold Analysis

## 1. Objective

Investigate how changing the classification threshold affects Logistic Regression performance in customer churn prediction.

## 2. Experimental Setup

- Dataset: Telco Customer Churn
- Training/testing split: 80% / 20%
- Random state: 42
- Model: Logistic Regression with StandardScaler
- Thresholds tested: 0.3, 0.4, 0.5, and 0.6
- Metrics: Accuracy, Precision, Recall, and F1-score

## 3. Results

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.3 | 74.95% | 51.93% | 75.40% | 61.50% |
| 0.4 | 77.64% | 56.69% | 66.84% | 61.35% |
| 0.5 | 80.70% | 65.84% | 56.68% | 60.92% |
| 0.6 | 79.70% | 70.95% | 39.84% | 51.03% |

## 4. Interpretation

A lower threshold classifies more customers as likely to churn. This increases recall but can also increase false-positive predictions.

A higher threshold is more selective. At 0.6, precision reached 70.95%, but recall dropped to 39.84%.

The 0.3 threshold achieved the highest recall (75.40%) and F1-score (61.50%) among the tested values. The default 0.5 threshold achieved the highest accuracy (80.70%).

## 5. Conclusion

The best threshold depends on the business objective. If identifying as many customers at risk of churn as possible is the priority, 0.3 is a promising candidate. However, it produces more false positives than the default threshold.

The 0.3 threshold should be treated as a candidate for further validation, not a final production setting. Its performance should be assessed on separate validation data and considered alongside the cost of retention offers and missed churners.

## 6. Next Steps

- Compare false positives and false negatives across thresholds.
- Validate threshold selection before deployment.
- Keep accuracy, precision, recall, and F1-score in view when selecting the operating threshold.