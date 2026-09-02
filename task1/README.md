---
**Internship Task Report**
**Task No.:** 1
**Task Title:** Load and Explore a Dataset
**Date:** August 2026
**Tool / Notebook:** `task1.ipynb`

---

## 1. Objective

The objective of this task was to select a dataset, load it into a Python environment, and perform a comprehensive Exploratory Data Analysis (EDA). This includes understanding the structure of the data, computing summary statistics, identifying patterns, and producing meaningful visualizations to document initial observations.

---

## 2. Introduction

Exploratory Data Analysis is the first and most critical step in any machine learning or data science workflow. Before building any model, an analyst must understand the underlying data — its shape, distributions, missing values, and inter-feature relationships. Skipping this step frequently leads to poor model choices and misleading results.

For this task, the **Iris Flower Dataset** was selected. It is a well-known benchmark dataset in the machine learning community, originally compiled by statistician Ronald Fisher in 1936. Despite its simplicity, it offers a rich set of features for practice: balanced classes, no missing values, and a clear separability structure that makes EDA findings directly verifiable through later modelling tasks.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | Iris Flower Dataset |
| Source | `sklearn.datasets.load_iris()` |
| Number of Instances | 150 |
| Number of Features | 4 (all continuous) |
| Target Variable | Species (3 classes) |
| Class Labels | *Iris setosa*, *Iris versicolor*, *Iris virginica* |
| Missing Values | None |
| Class Balance | Perfectly balanced — 50 instances per class |

**Feature Descriptions:**

| Feature | Unit | Description |
|---|---|---|
| Sepal Length | cm | Length of the outer leaf-like parts of the flower |
| Sepal Width | cm | Width of the outer leaf-like parts |
| Petal Length | cm | Length of the inner flower petals |
| Petal Width | cm | Width of the inner flower petals |

---

## 4. Methodology

### 4.1 Data Loading
The dataset was loaded using scikit-learn's built-in loader and converted into a `pandas DataFrame` for ease of manipulation:

```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
```

### 4.2 Initial Inspection
- `df.head()` was used to preview the first five rows and verify column structure.
- `df.isnull().sum()` confirmed zero missing values across all features.
- `df['species'].value_counts()` confirmed a perfectly balanced class distribution (50 per class).

### 4.3 Summary Statistics
`df.describe()` was applied to obtain per-feature statistics including mean, standard deviation, minimum, maximum, and quartile values. Additionally, a `groupby('species').mean()` was computed to identify class-level differences.

### 4.4 Visualizations

Three visualizations were produced:

1. **Histograms** (one per feature, overlaid by species): Used to understand the distribution of each feature and how well each feature separates the classes.
2. **Scatter Plot** (Petal Length vs. Petal Width): The most informative two-feature view, used to assess linear separability between species.
3. **Correlation Heatmap**: A Pearson correlation matrix to identify redundant or strongly related features.

---

## 5. Results and Analysis

### 5.1 Summary Statistics (Per Class)

| Feature | Setosa | Versicolor | Virginica |
|---|---|---|---|
| Sepal Length (cm) | 5.01 | 5.94 | 6.59 |
| Sepal Width (cm) | 3.43 | 2.77 | 2.97 |
| Petal Length (cm) | 1.46 | 4.26 | 5.55 |
| Petal Width (cm) | 0.25 | 1.33 | 2.03 |

The difference in petal measurements across species is substantially larger than in sepal measurements. This suggests petal features carry more discriminative power for classification.

### 5.2 Correlation Analysis

| Feature Pair | Pearson r |
|---|---|
| Petal Length & Petal Width | +0.96 |
| Sepal Length & Petal Length | +0.87 |
| Sepal Width & Petal Width | -0.37 |

Petal length and petal width are highly correlated (r ≈ 0.96), indicating multicollinearity. These two features essentially carry the same information. Sepal width shows weak or slightly negative correlations with the other features, making it the least informative feature for classification.

### 5.3 Key Observations

- *Iris setosa* is **completely linearly separable** from the other two species using petal measurements alone. A simple threshold on petal length (< 2 cm) identifies setosa with 100% accuracy.
- *Iris versicolor* and *Iris virginica* exhibit some overlap in the petal feature space, requiring more sophisticated classifiers to distinguish reliably.
- Sepal width shows the widest distribution for *setosa* but overlaps heavily for *versicolor* and *virginica*, making it a weak stand-alone predictor.

---

## 6. Conclusion

The Exploratory Data Analysis revealed that the Iris dataset is well-structured, clean, and exhibits clear patterns useful for classification. The petal features (length and width) are the most informative, while sepal features — particularly sepal width — contribute less. These findings will directly inform feature selection decisions in subsequent modelling tasks (Tasks 2, 4, 5).

The visualizations produced confirm that at least one class (*setosa*) is trivially separable, while the remaining two require a proper classifier, setting realistic expectations for model performance in future tasks.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task1.ipynb` | Jupyter notebook with all code and inline outputs |
| `task1_histograms.png` | Feature distributions by species |
| `task1_scatter.png` | Petal length vs petal width scatter |
| `task1_correlation.png` | Pearson correlation heatmap |

---

## 8. Tools and Libraries

| Library | Version | Purpose |
|---|---|---|
| Python | 3.14 | Programming language |
| pandas | latest | Data loading and manipulation |
| numpy | latest | Numerical operations |
| matplotlib | latest | Base visualisation |
| seaborn | latest | Statistical visualisation (heatmap) |
| scikit-learn | latest | Dataset loader |
