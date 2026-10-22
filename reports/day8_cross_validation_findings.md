# Day 8 — Cross-Validation Model Evaluation

## 1. Objective

Evaluate Logistic Regression and Decision Tree using 5-fold stratified cross-validation to compare their performance across multiple data splits.

## 2. Experimental Setup

- Dataset: Telco Customer Churn
- Total records: 7,043
- Cross-validation method: Stratified 5-fold cross-validation
- Shuffling enabled: Yes
- Random state: 42
- Evaluation metrics: Accuracy, Precision, Recall, and F1-score

## 3. Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.52% | 65.80% | 55.21% | 60.03% |
| Decision Tree | 79.09% | 62.89% | 51.79% | 56.71% |

## 4. Interpretation

### Logistic Regression

Logistic Regression achieved an average accuracy of 80.52% and an F1-score of 60.03%. Its precision was 65.80%, while its recall was 55.21%.

### Decision Tree

The tuned Decision Tree achieved an average accuracy of 79.09% and an F1-score of 56.71%. Its precision was 62.89%, and its recall was 51.79%.

## 5. Comparison and Conclusion

Logistic Regression performed better than the tuned Decision Tree across all four evaluation metrics.

The results are consistent with the earlier train-test evaluation, where Logistic Regression also outperformed the Decision Tree.

Cross-validation provides a more robust estimate of performance than relying on a single split. However, the current script reports average scores only; calculating the standard deviation across folds would help assess performance variability.

## 6. Next Steps

- Examine performance variation across folds.
- Continue improving churn detection, particularly recall.
- Keep model selection based on appropriate validation and evaluation rather than accuracy alone.