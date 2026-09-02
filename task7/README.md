---
**Internship Task Report**
**Task No.:** 7
**Task Title:** Implement Principal Component Analysis (PCA)
**Date:** August 2026
**Tool / Notebook:** `task7.ipynb`

---

## 1. Objective

The objective of this task was to apply Principal Component Analysis (PCA) to a high-dimensional dataset, analyse the explained variance distribution across principal components, determine the minimum number of components required to retain meaningful information, and visualise the data in reduced-dimensional space.

---

## 2. Introduction

Principal Component Analysis is a linear dimensionality reduction technique that transforms a set of correlated variables into a smaller set of uncorrelated variables called **principal components**. Each principal component is a linear combination of the original features, oriented in the direction of maximum variance in the data.

Mathematically, PCA computes the eigenvectors and eigenvalues of the feature covariance matrix. The eigenvectors define the directions (principal components), and the eigenvalues define the amount of variance captured along each direction. By retaining only the top-k eigenvectors, the data is projected into a k-dimensional subspace that preserves the maximum possible variance.

PCA serves several practical purposes:
- **Dimensionality reduction** for computational efficiency
- **Noise filtering** (low-variance components often represent noise)
- **Visualisation** of high-dimensional data in 2D or 3D
- **Multicollinearity removal** for downstream modelling

This task applied PCA to the **Wine dataset**, which contains 13 chemical measurements for 178 wine samples from three different cultivar classes — a suitable dataset for demonstrating PCA's ability to compress high-dimensional chemical data while preserving class-discriminative structure.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | Wine Recognition Dataset |
| Source | `sklearn.datasets.load_wine()` |
| Number of Instances | 178 |
| Number of Features | 13 (all continuous, chemical measurements) |
| Target Variable | Cultivar class (3 classes) |
| Missing Values | None |

**Features include:** alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, colour intensity, hue, OD280/OD315 of diluted wines, proline.

---

## 4. Methodology

### 4.1 Data Standardisation
PCA is scale-sensitive. If features have vastly different ranges, the principal components will be dominated by high-magnitude features. Standardisation to zero mean and unit variance ensures that PCA treats all features equitably:

```python
X_scaled = StandardScaler().fit_transform(X)
```

This step is non-negotiable in PCA workflows. For example, the `proline` feature has values in the range 278–1680, while `nonflavanoid phenols` ranges from 0.13 to 0.66. Without scaling, `proline` alone would dominate the first principal component.

### 4.2 Full PCA for Variance Analysis
PCA was first run retaining all 13 components to compute the full explained variance distribution:

```python
pca_full = PCA()
pca_full.fit(X_scaled)
```

The **scree plot** (bar chart of per-component explained variance) and **cumulative explained variance curve** were generated to identify the optimal number of components to retain.

### 4.3 2D Projection
A second PCA with `n_components=2` was fitted to project the data onto the first two principal components for 2D scatter visualisation. Each point was coloured by cultivar class (without the PCA algorithm having access to this information).

### 4.4 3D Projection
A third PCA with `n_components=3` was fitted for a 3D scatter visualisation using matplotlib's 3D axes, offering additional separation context beyond the 2D view.

### 4.5 Loading Analysis
The **loading matrix** (`pca.components_.T`) was extracted to identify which original features contribute most to the first two principal components. This provides interpretability to the otherwise abstract reduced-dimensional space.

---

## 5. Results and Analysis

### 5.1 Explained Variance by Component

| Principal Component | Individual Variance (%) | Cumulative Variance (%) |
|---|---|---|
| PC1 | ~36.2 | ~36.2 |
| PC2 | ~19.2 | ~55.4 |
| PC3 | ~11.1 | ~66.5 |
| PC4 | ~7.3 | ~73.8 |
| PC5 | ~6.6 | ~80.4 |
| PC6 | ~5.4 | ~85.8 |
| PC7 | ~3.4 | ~89.2 |
| 13 (all) | — | 100.0 |

**Key thresholds:**
- 90% variance retained with **7 components** (down from 13 — a 46% reduction)
- 95% variance retained with approximately 9 components

### 5.2 2D Projection Results
Despite capturing only ~55.4% of total variance, the 2D projection (PC1 vs. PC2) shows **clear and nearly complete separation** of the three wine cultivar classes. This is a significant finding — the chemical composition differences between cultivars represent the dominant source of variation in this dataset, such that even a 2-component projection preserves the class structure.

### 5.3 3D Projection Results
The 3D projection (PC1 vs. PC2 vs. PC3, ~66.5% variance) shows further improved separation between the three classes, with fewer overlapping regions compared to the 2D view.

### 5.4 Loading Analysis (PC1)
PC1 has high positive loadings on flavanoids, total phenols, OD280/OD315 ratio, and hue, and high negative loadings on colour intensity and malic acid. This suggests that PC1 broadly captures a contrast between **phenolic richness** (associated with red wine quality) and **acidity/colour intensity** — a chemically meaningful axis.

---

## 6. Conclusion

PCA successfully reduced the 13-dimensional Wine dataset to 2 components that visually reveal the three-class cultivar structure. The analysis demonstrated that the first 7 principal components account for over 90% of total variance, enabling significant dimensionality reduction with minimal information loss. The unsupervised projection naturally separated the three cultivar classes because the chemical features that differ most across samples are precisely those that distinguish the cultivars. The loading analysis added interpretability, mapping the abstract PC1 axis to a meaningful contrast between phenolic and acidic chemical properties.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task7.ipynb` | Jupyter notebook with all code and inline outputs |
| `task7_variance.png` | Scree plot and cumulative explained variance curve |
| `task7_pca_2d.png` | 2D scatter plot (PC1 vs. PC2), coloured by cultivar |
| `task7_pca_3d.png` | 3D scatter plot (PC1 vs. PC2 vs. PC3) |

---

## 8. Tools and Libraries

| Library | Purpose |
|---|---|
| scikit-learn | PCA, StandardScaler, load_wine |
| matplotlib, mpl_toolkits | 2D and 3D visualisation |
| numpy, pandas | Data handling and loading matrix construction |
