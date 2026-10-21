# Day 7 — Decision Tree Hyperparameter Tuning

## 1. Objective

Test different Decision Tree configurations to improve customer churn prediction and compare their results with the Logistic Regression baseline.

## 2. Experiment Setup

- Dataset: IBM Telco Customer Churn
- Training records: 5,634
- Testing records: 1,409
- Test split: 20%
- Random state: 42
- Configurations tested: 5

## 3. Results

| Max depth | Minimum leaf samples | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 78.71% | 66.82% | 39.30% | 49.49% |
| 5 | 1 | 79.42% | 63.12% | 54.01% | 58.21% |
| 7 | 1 | 78.78% | 61.83% | 52.41% | 56.73% |
| 10 | 5 | 75.66% | 54.57% | 49.47% | 51.89% |
| Unlimited | 10 | 77.29% | 59.12% | 46.79% | 52.24% |

## 4. Best Configuration Tested

The Decision Tree with `max_depth=5` and `min_samples_leaf=1` achieved the highest F1-score among the five configurations tested.

- Accuracy: 79.42%
- Precision: 63.12%
- Recall: 54.01%
- F1-score: 58.21%

## 5. Comparison with Logistic Regression

The Logistic Regression baseline achieved:

- Accuracy: 80.70%
- Precision: 65.84%
- Recall: 56.68%
- F1-score: 60.92%

Logistic Regression performed better on all four metrics in this experiment.

## 6. Conclusion

The Decision Tree tuning experiment showed that tree depth and minimum leaf size affect churn prediction performance. Among the tested configurations, the depth-5 tree achieved the strongest overall balance of precision and recall.

However, the tuned Decision Tree did not outperform Logistic Regression on the current test set. Additional experiments, such as cross-validated hyperparameter tuning and threshold adjustment, may help improve churn detection.

These findings apply to the tested configurations and split; further validation is needed before selecting a production model.