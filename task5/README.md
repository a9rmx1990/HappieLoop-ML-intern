---
**Internship Task Report**
**Task No.:** 5
**Task Title:** Build a Decision Tree Classifier
**Date:** August 2026
**Tool / Notebook:** `task5.ipynb`

---

## 1. Objective

The objective of this task was to implement a Decision Tree classifier to predict categorical outcomes, systematically evaluate the effect of tree depth on model generalisation, and interpret the resulting tree structure and feature importance scores.

---

## 2. Introduction

A Decision Tree is a supervised learning algorithm that partitions the feature space into rectangular regions through a series of binary splits. At each internal node, the algorithm selects the feature and threshold that maximally reduces impurity in the resulting subsets. The most commonly used impurity measures are **Gini impurity** and **entropy (information gain)**.

Decision Trees are among the most interpretable machine learning models. Unlike black-box methods such as neural networks, every prediction made by a decision tree can be traced through an explicit sequence of human-readable rules. This interpretability makes them particularly valuable in regulated domains — healthcare, finance, and legal systems — where model decisions must be auditable and explainable.

This task applied a Decision Tree classifier to the Iris flower dataset. The key focus was on the bias-variance tradeoff through a depth-sweep experiment, followed by training and evaluating the optimal model.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | Iris Flower Dataset |
| Source | `sklearn.datasets.load_iris()` |
| Number of Instances | 150 |
| Number of Features | 4 (sepal length, sepal width, petal length, petal width) |
| Target Variable | Species (3 classes: setosa, versicolor, virginica) |
| Train/Test Split | 75% training / 25% test (stratified) |
| Training Instances | 112 |
| Test Instances | 38 |

Stratified splitting was applied to ensure proportional class representation in both subsets.

---

## 4. Methodology

### 4.1 Depth-Sweep Experiment
Before training the final model, a systematic sweep was conducted across tree depths from 1 to 9. For each depth, a `DecisionTreeClassifier` was trained on the training set and evaluated on both the training and test sets. This experiment was designed to directly observe the **bias-variance tradeoff**:

- Low depth → high bias, underfitting
- High depth → low bias, high variance, overfitting

The results were plotted as a dual-line chart (training accuracy vs. test accuracy as a function of depth).

### 4.2 Model Selection
Based on the depth-sweep results, `max_depth = 3` was selected as the optimal hyperparameter. At this depth, test accuracy approaches its maximum and the gap between training and test accuracy remains small (minimal overfitting).

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
```

### 4.3 Model Evaluation
The following metrics were computed on the test set:
- **Accuracy:** Overall proportion of correctly classified instances
- **Precision, Recall, F1-score:** Per-class metrics from the classification report
- **Confusion Matrix:** Visual display of correct and incorrect classifications per class

### 4.4 Tree Visualisation
The fitted decision tree was visualised using `sklearn.tree.plot_tree`, producing a diagram that shows each split condition, Gini impurity, sample counts, and class distribution at every node. A text representation was also produced using `export_text`.

### 4.5 Feature Importance Analysis
Scikit-learn computes feature importance as the total weighted Gini impurity reduction attributable to each feature across all splits. These values were plotted as a horizontal bar chart.

---

## 5. Results and Analysis

### 5.1 Depth-Sweep Analysis

| max_depth | Training Accuracy | Test Accuracy |
|---|---|---|
| 1 | ~66% | ~66% |
| 2 | ~96% | ~92% |
| **3** | **~97%** | **~97%** |
| 4 | ~99% | ~97% |
| 5+ | 100% | ~97% (plateaus) |

At depth 3, training and test accuracy are nearly equal (~97%), indicating an effective balance between model complexity and generalisation. Beyond depth 4, training accuracy continues to rise toward 100% while test accuracy does not improve — a classic signature of overfitting.

### 5.2 Classification Report (max_depth = 3)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Setosa | 1.00 | 1.00 | 1.00 | 13 |
| Versicolor | 0.94 | 1.00 | 0.97 | 13 |
| Virginica | 1.00 | 0.92 | 0.96 | 12 |
| **Overall** | | | **~0.97** | 38 |

**Overall Test Accuracy: ~97.37%**

Setosa achieved perfect classification across precision and recall. Minor confusion exists between Versicolor and Virginica — consistent with the biological overlap observed in the EDA (Task 1) and clustering (Task 4) tasks.

### 5.3 Decision Tree Structure

The first split at the root node is:
```
petal width (cm) ≤ 0.80  →  Pure Setosa leaf (Gini = 0.00)
petal width (cm) > 0.80  →  Continue to depth 2
```

A single threshold on petal width perfectly separates Setosa from all other samples. Subsequent splits on petal width and petal length resolve the Versicolor/Virginica boundary.

### 5.4 Feature Importances

| Feature | Gini Importance |
|---|---|
| Petal Width | ~0.92 |
| Petal Length | ~0.06 |
| Sepal Length | ~0.02 |
| Sepal Width | 0.00 |

Petal width is overwhelmingly the most important feature, contributing ~92% of the total impurity reduction. This finding is consistent across all tasks in this internship that have used the Iris dataset (EDA, K-Means, and now Decision Tree).

---

## 6. Conclusion

The Decision Tree classifier achieved approximately 97% accuracy on the Iris test set using a tree of depth 3. The depth-sweep experiment provided a clear demonstration of the bias-variance tradeoff, confirming that trees deeper than 3 levels overfit without improving generalisation. The resulting tree is highly interpretable — a single petal width threshold separates *setosa* perfectly, and two additional splits resolve the remaining boundary between *versicolor* and *virginica*. Feature importance analysis confirmed that petal width and petal length are the primary discriminative features, with sepal width contributing nothing to classification performance.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task5.ipynb` | Jupyter notebook with all code and inline outputs |
| `task5_depth_sweep.png` | Training vs test accuracy across tree depths 1–9 |
| `task5_confusion.png` | Confusion matrix heatmap on test set |
| `task5_tree.png` | Full visual decision tree diagram (max_depth=3) |
| `task5_importances.png` | Feature importance bar chart |

---

## 8. Tools and Libraries

| Library | Purpose |
|---|---|
| scikit-learn | DecisionTreeClassifier, plot_tree, export_text, metrics |
| seaborn | Confusion matrix heatmap |
| matplotlib | All plots |
| pandas, numpy | Data handling |
