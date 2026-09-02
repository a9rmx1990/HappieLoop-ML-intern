---
**Internship Task Report**
**Task No.:** 10
**Task Title:** Binary Classification with Support Vector Machine (SVM)
**Date:** August 2026
**Tool / Notebook:** `task10.ipynb`

---

## 1. Objective

The objective of this task was to implement a Support Vector Machine (SVM) for binary classification on a high-stakes biomedical dataset, systematically compare linear and non-linear kernel functions using 5-fold cross-validation, conduct hyperparameter optimization using Grid Search, evaluate model performance using comprehensive diagnostic metrics (Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC), and interpret feature importances.

---

## 2. Introduction

Support Vector Machines (SVMs) are among the most robust and mathematically grounded supervised learning algorithms for binary and multiclass classification. The fundamental concept of an SVM is to find the **maximum margin hyperplane** — a decision boundary that maximizes the geometric distance (margin) between the separating hyperplane and the closest training instances from either class (known as **support vectors**).

Key strengths of the SVM architecture:
- **Maximum Margin Principle:** Maximizing the margin provides strong theoretical bounds on generalization error, making SVMs resilient to overfitting in high-dimensional spaces.
- **The Kernel Trick:** For datasets that are not linearly separable in their original input space, kernel functions ($K(x_i, x_j)$) implicitly map input vectors into higher-dimensional feature spaces where a linear separation becomes possible, without explicitly computing coordinates in that high-dimensional space.
- **Regularization Parameter ($C$):** Governs the trade-off between maximizing the margin and minimizing classification errors on the training set (soft-margin formulation).

In this task, an SVM was developed to classify breast tumors as either **malignant** or **benign** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. In medical diagnostics, classification accuracy alone is inadequate — sensitivity (recall) on the malignant class is of paramount clinical importance to minimize life-threatening false negatives.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | `sklearn.datasets.load_breast_cancer()` |
| Number of Instances | 569 |
| Number of Features | 30 continuous features computed from digitized FNA images |
| Target Variable | Diagnosis (`0 = Malignant`, `1 = Benign`) |
| Class Distribution | 212 Malignant (37.3%), 357 Benign (62.7%) |
| Missing Values | None |
| Train / Test Split | 80% Training (455 samples) / 20% Testing (114 samples), Stratified |

The 30 numerical features capture morphological properties of cell nuclei present in fine needle aspirate (FNA) samples, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (computed across mean, standard error, and "worst"/largest values).

---

## 4. Methodology

### 4.1 Data Standardization
SVMs calculate geometric Euclidean distances between support vectors and decision boundaries. If features differ in scale (e.g., `area_mean` ranging from 100 to 2500 vs. `smoothness_mean` ranging from 0.05 to 0.16), the distance metric is completely dominated by high-magnitude features.

Standardization was strictly applied using `StandardScaler`, fit exclusively on training data and applied to both training and test sets:
```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

### 4.2 Kernel Function Exploration
To evaluate how different mathematical kernels model the decision surface, four kernel functions were compared using **5-fold stratified cross-validation** on the training data:
1. **Linear Kernel:** $K(x, x') = \langle x, x' \rangle$
2. **Radial Basis Function (RBF) Kernel:** $K(x, x') = \exp(-\gamma \|x - x'\|^2)$
3. **Polynomial Kernel:** $K(x, x') = (\gamma \langle x, x' \rangle + r)^d$
4. **Sigmoid Kernel:** $K(x, x') = \tanh(\gamma \langle x, x' \rangle + r)$

Results were visualised via box plots (`task10_kernels.png`) using `plt.boxplot(..., tick_labels=kernels)`.

### 4.3 Hyperparameter Optimization (Grid Search)
Focusing on the top-performing RBF kernel, a 5-fold cross-validated grid search (`GridSearchCV`) was executed to find the optimal combination of:
- **$C$ (Regularization parameter):** `[0.1, 1, 10, 100]`
- **$\gamma$ (RBF kernel bandwidth parameter):** `['scale', 'auto', 0.01, 0.001]`

### 4.4 Final Model Evaluation
The best estimator was evaluated on the unseen test set ($N = 114$) using:
- Overall Accuracy
- Precision, Recall, and F1-Score per class
- Confusion Matrix Heatmap
- Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC)

### 4.5 Feature Importance Analysis (Linear SVM)
While non-linear kernels do not expose direct feature weights, a Linear SVM ($C=1$) was trained to inspect the magnitude of the learned weight vector coefficients ($\beta$), highlighting the 10 most influential cellular characteristics.

---

## 5. Results and Analysis

### 5.1 Kernel Comparison (5-Fold Cross-Validation)

| Kernel Type | Mean CV Accuracy | Standard Deviation | Assessment |
|---|---|---|---|
| **RBF (Radial Basis)** | **~97.8%** | **±1.1%** | **Best overall performance and stability** |
| Linear | ~96.9% | ±1.3% | Strong linear separability in scaled 30D space |
| Polynomial | ~91.2% | ±2.4% | Moderate performance; prone to boundary artifacts |
| Sigmoid | ~95.2% | ±1.8% | Sensitive to scaling and parameter mismatch |

The RBF kernel demonstrated the highest mean accuracy and lowest variance across folds.

### 5.2 Grid Search Optimization

| Parameter | Optimal Value | Description |
|---|---|---|
| `kernel` | `'rbf'` | Radial Basis Function |
| `C` | `10` | Moderate-high penalty for margin violations |
| `gamma` | `'scale'` | Normalized bandwidth based on feature dimension |
| **Best CV Score** | **~98.2%** | Mean 5-fold cross-validation accuracy |

### 5.3 Test Set Classification Metrics

```
              precision    recall  f1-score   support

   malignant       0.98      0.95      0.96        42
      benign       0.97      0.99      0.98        72

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

- **Test Accuracy:** **97.37%** (111 / 114 correct classifications)
- **Malignant Recall (Sensitivity):** **95.24%** (40 / 42 malignant tumors correctly detected)
- **Benign Precision:** **97.26%**
- **ROC-AUC Score:** **0.996** (Near-perfect discrimination across all decision thresholds)

### 5.4 Confusion Matrix & ROC Curve Analysis
- **False Negatives:** Only 2 malignant cases out of 42 were misclassified as benign.
- **False Positives:** Only 1 benign case out of 72 was misclassified as malignant.
- The ROC curve (`task10_eval.png`) rises almost vertically toward the upper-left coordinate $(0, 1)$, demonstrating high true positive rates at very low false positive thresholds.

### 5.5 Most Influential Features (Linear SVM Coefficients)
The top 5 morphological features driving malignant classifications were:
1. `worst texture` — Heterogeneity and irregularity of nuclear surface
2. `worst concavity` — Severity of concave portions of the nuclear contour
3. `worst perimeter` — Total outer boundary length of largest nuclei
4. `worst radius` — Distance from center to perimeter points
5. `mean concavity` — Average depth of contour indentations

These quantitative indicators strongly align with established clinical cytopathology: enlarged, irregular, and deeply indented cell nuclei are hallmarks of malignant tissue growth.

---

## 6. Key Takeaways and Clinical Relevance

- **Feature scaling is foundational:** Without `StandardScaler`, distance-based classifiers like SVM fail completely on multi-scale biological measurements.
- **Threshold tuning in healthcare:** In clinical deployments, probability thresholds can be adjusted to prioritize recall on the malignant class (reducing false negatives to zero) at the expense of a slight increase in confirmatory biopsies (false positives).
- **Linear vs Non-Linear trade-off:** While the RBF kernel yielded marginally superior accuracy (97.8% vs 96.9%), the Linear kernel provided direct feature coefficient interpretability, which is vital for clinical validation.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task10.ipynb` | Jupyter notebook containing the complete pipeline, execution, and outputs |
| `task10_kernels.png` | 5-fold cross-validation accuracy box plots across SVM kernels |
| `task10_eval.png` | Dual evaluation panel: Confusion Matrix heatmap and ROC Curve (AUC = 0.996) |
| `task10_features.png` | Horizontal bar chart of top 10 most influential features from Linear SVM |

---

## 8. Tools and Libraries

| Library | Purpose |
|---|---|
| scikit-learn | `load_breast_cancer`, `StandardScaler`, `SVC`, `GridSearchCV`, `cross_val_score`, `metrics` |
| seaborn | Confusion matrix visualization |
| matplotlib | Kernel comparison, ROC curve, and feature importance bar charts |
| pandas, numpy | Dataset manipulation, array operations, coefficient ranking |
