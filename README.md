# End-to-End Machine Learning Pipeline

This project demonstrates a complete, production-ready machine learning workflow
using scikit-learn Pipeline and ColumnTransformer on the Breast Cancer dataset.

It includes:
- Preprocessing (scaling)
- Model training
- Evaluation
- Visualization
- Model persistence for deployment

---

## 🔍 Objective

Build a reusable machine learning pipeline that integrates preprocessing and
classification into a single workflow, ensuring:
- No data leakage
- Consistent preprocessing
- Easy deployment

---

## 📊 Dataset

We use the **Breast Cancer dataset** from `scikit-learn`:

- Binary classification (malignant vs benign)
- Only numerical features
- Clean and balanced dataset

---

## 🛠️ Workflow

1. Load dataset  
2. Stratified train-test split  
3. Preprocess with `StandardScaler` using `ColumnTransformer`  
4. Build full `Pipeline` with scaling + model  
5. Train using Logistic Regression  
6. Evaluate metrics  
7. Plot confusion matrix  
8. Save full pipeline using joblib

---

## 📈 Evaluation Metrics

The pipeline was evaluated using:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**

The results show the model performs strongly across all metrics.

---

## 📌 Confusion Matrix Analysis

The confusion matrix (below) summarizes test performance:

| Actual / Predicted | 0 (malignant) | 1 (benign) |
|-------------------|---------------|------------|
| 0 (malignant)     | 41            | 1          |
| 1 (benign)        | 1             | 71         |

This means:
- True Negatives (TN): 41
- False Positives (FP): 1
- False Negatives (FN): 1
- True Positives (TP): 71

Derived metrics:
- **Accuracy ≈ 98.2%**
- **Precision ≈ 98.6%**
- **Recall ≈ 98.6%**
- **F1-Score ≈ 98.6%**

Minimal misclassifications show the model generalizes well.

---

## 📷 Confusion Matrix Visualization

![Confusion Matrix](Confusion_matrix.png)

---

## 📂 Files in This Repository

| File | Description |
|------|-------------|
| `End_to_End_ML_Pipeline.ipynb` | Full notebook with explanations |
| `ml_pipeline.pkl` | Saved complete pipeline ready for deployment |
| `Confusion_matrix.png` | PNG image of confusion matrix |
| `README.md` | Project documentation |

---

## 🛠️ Tools Used

- Python  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- joblib

---

## 📌 Key Takeaways

- Using a pipeline simplifies preprocessing + modeling.
- Confusion matrix provides insight into classification errors.
- Saved pipeline ensures consistency when deployed.
- This structure is deployment-ready and professional.
