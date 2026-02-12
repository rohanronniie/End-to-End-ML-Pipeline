# End-to-End Machine Learning Pipeline

This project demonstrates a complete, production-ready
machine learning workflow using:

- ColumnTransformer
- Pipeline
- Logistic Regression
- Model Evaluation
- Model Persistence

---

## Objective

To build a reusable machine learning pipeline
that integrates preprocessing and model training
into a single workflow.

---

## Dataset

Breast Cancer dataset from scikit-learn.

- Binary classification problem
- Numerical medical features
- Predicts malignant vs benign tumors

---

## Workflow

1. Load dataset
2. Perform stratified train-test split
3. Apply preprocessing using ColumnTransformer
4. Build full ML pipeline
5. Train model
6. Evaluate using accuracy, precision, recall, F1-score
7. Visualize confusion matrix
8. Save complete pipeline using joblib

---

## Why Pipeline?

Pipelines:

- Prevent data leakage
- Ensure consistent preprocessing
- Simplify deployment
- Improve reproducibility

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Confusion Matrix Analysis

The confusion matrix shows model prediction performance:

|                | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| Actual 0      | 41          | 1           |
| Actual 1      | 1           | 71          |

This means:

- True Negatives (TN): 41
- False Positives (FP): 1
- False Negatives (FN): 1
- True Positives (TP): 71

Only 2 misclassifications out of 114 test samples.

### Derived Metrics

- Accuracy ≈ 98.2%
- Precision ≈ 98.6%
- Recall ≈ 98.6%
- F1-Score ≈ 98.6%

The model demonstrates excellent generalization
with minimal false predictions.

---

## Confusion Matrix Visualization

![Confusion Matrix](Confusion_matrix.png)
---

## Files in Repository

| File | Description |
|------|------------|
| End_to_End_ML_Pipeline.ipynb | Complete implementation |
| ml_pipeline.pkl | Saved full pipeline |
| README.md | Project documentation |

---

## Tools Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
