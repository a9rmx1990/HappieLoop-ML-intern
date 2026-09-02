---
**Internship Task Report**
**Task No.:** 4
**Task Title:** Implement a K-Means Clustering Algorithm
**Date:** August 2026
**Tool / Notebook:** `task4.ipynb`

---

## 1. Objective

The objective of this task was to implement and evaluate a K-Means clustering algorithm on a real dataset, select the optimal number of clusters using quantitative methods, and compare the discovered cluster structure against ground-truth labels to assess clustering quality.

---

## 2. Introduction

Clustering is an unsupervised machine learning technique that partitions data into groups (clusters) such that instances within the same group are more similar to each other than to those in other groups. Unlike supervised classification, clustering operates without access to class labels — it discovers structure purely from the input features.

K-Means is one of the most widely used clustering algorithms due to its simplicity, scalability, and interpretability. Given a predefined number of clusters K, it iteratively assigns each data point to the nearest centroid and updates centroids as the mean of their assigned points, converging to a local minimum of the within-cluster sum of squares (inertia).

In this task, K-Means was applied to the Iris flower dataset, with species labels intentionally withheld from the algorithm to simulate a real unsupervised scenario. The quality of the resulting clusters was then assessed by comparing them against the true species labels.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | Iris Flower Dataset |
| Source | `sklearn.datasets.load_iris()` |
| Number of Instances | 150 |
| Features Used | 4 (sepal length, sepal width, petal length, petal width) |
| Labels | Withheld during training; used only for evaluation |
| True Number of Classes | 3 (*setosa*, *versicolor*, *virginica*) |

---

## 4. Methodology

### 4.1 Data Standardisation
K-Means relies on Euclidean distance as its similarity metric. Features with larger numerical ranges disproportionately influence distance calculations, which can cause the algorithm to ignore informative features with smaller ranges. To address this, all features were standardised using `StandardScaler` prior to clustering:

```python
X_scaled = StandardScaler().fit_transform(X)
```

Post-scaling, each feature has a mean of 0 and standard deviation of 1.

### 4.2 Optimal K Selection

Two complementary methods were used to determine the appropriate number of clusters:

**Elbow Method:**
K-Means was run for K = 1 through K = 10. The inertia (within-cluster sum of squared distances) was recorded for each K. The optimal K corresponds to the point where the inertia curve exhibits a pronounced "elbow" — the rate of improvement diminishes sharply beyond that point.

**Silhouette Score:**
The silhouette score measures how similar each point is to its own cluster compared to the nearest neighbouring cluster. Values range from -1 (misclassified) to +1 (well-clustered). The K with the highest silhouette score is considered optimal.

Both methods independently identified **K = 3** as the optimal number of clusters.

### 4.3 Final Model Training
K-Means was trained with K = 3, `n_init=10` (10 random initialisations to avoid local minima), and `random_state=42` for reproducibility.

### 4.4 Evaluation
Two external validation metrics were computed by comparing cluster assignments to the true species labels:

- **Silhouette Score:** Measures internal cluster quality (no labels required)
- **Adjusted Rand Index (ARI):** Measures agreement between cluster assignments and true labels, corrected for chance. ARI = 1.0 indicates perfect agreement; ARI = 0.0 indicates random labelling.

---

## 5. Results and Analysis

### 5.1 K Selection Results

| K | Silhouette Score |
|---|---|
| 2 | ~0.49 |
| **3** | **~0.55 (peak)** |
| 4 | ~0.49 |
| 5 | ~0.45 |

The elbow in the inertia curve and the peak silhouette score both occur at K = 3, confirming the choice.

### 5.2 Clustering Quality Metrics

| Metric | Value | Interpretation |
|---|---|---|
| Silhouette Score | ~0.55 | Reasonably well-separated, moderately dense clusters |
| Adjusted Rand Index | ~0.73 | Strong agreement with true species labels |

An ARI of 0.73 signifies that K-Means recovered most of the ground-truth species structure without ever observing the labels.

### 5.3 Cluster vs. True Label Comparison

When visualised on petal length vs. petal width (the most discriminative feature pair):
- The cluster assigned to *Iris setosa* is **perfectly isolated** — zero overlap with the other two clusters.
- The clusters for *Iris versicolor* and *Iris virginica* exhibit **minor boundary overlap**, consistent with the EDA findings from Task 1.

The cluster centroids, when inverse-transformed back to the original feature scale, closely correspond to the per-species feature means computed in Task 1, further validating the clustering result.

---

## 6. Conclusion

The K-Means clustering algorithm successfully recovered the three species groupings present in the Iris dataset without access to any labels. The use of feature standardisation was critical to obtaining meaningful clusters. Both quantitative selection methods (elbow and silhouette) converged on K = 3, and the resulting clusters achieved an Adjusted Rand Index of ~0.73 against the ground truth.

The experiment confirms a fundamental insight: when the natural structure of the data aligns with the true categories, unsupervised methods can discover meaningful groupings autonomously. The minor overlap between *versicolor* and *virginica* clusters reflects a genuine biological similarity between these two species that cannot be eliminated by the algorithm itself.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task4.ipynb` | Jupyter notebook with all code and inline outputs |
| `task4_elbow.png` | Inertia vs. K (elbow plot) and silhouette score vs. K |
| `task4_clusters.png` | Side-by-side: K-Means clusters vs. true species labels |

---

## 8. Tools and Libraries

| Library | Purpose |
|---|---|
| scikit-learn | KMeans, StandardScaler, silhouette_score, adjusted_rand_score |
| matplotlib | Visualisation |
| numpy | Array operations |
