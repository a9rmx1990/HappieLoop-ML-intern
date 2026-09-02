---
**Internship Task Report**
**Task No.:** 2
**Task Title:** Implement a Simple Linear Regression Model
**Date:** August 2026
**Tool / Notebook:** `task2.ipynb`

---

## 1. Objective

The objective of this task was to build a simple linear regression model using a real-world dataset, evaluate its predictive performance using standard regression metrics, and visually analyse the model fit and residual behaviour.

---

## 2. Introduction

Linear regression is one of the foundational algorithms in machine learning and statistics. It models the relationship between one or more predictor variables and a continuous target variable by fitting a straight line (or hyperplane) that minimises the sum of squared prediction errors — a method known as Ordinary Least Squares (OLS).

The mathematical form of simple linear regression (one predictor) is:

```
ŷ = β₀ + β₁x
```

Where:
- `ŷ` is the predicted value
- `β₀` is the intercept
- `β₁` is the coefficient (slope) for predictor `x`

This task applies simple linear regression to predict housing prices from a single economic indicator — median income — using the California Housing Dataset. The goal is not to build the best possible model, but to understand the mechanics of regression and interpret the outputs meaningfully.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | California Housing Dataset |
| Source | `sklearn.datasets.fetch_california_housing()` |
| Number of Instances | 20,640 |
| Number of Features | 8 (only 1 used in this task) |
| Predictor Used | `MedInc` — median income of block group (scaled units) |
| Target Variable | `MedHouseVal` — median house value (in units of $100,000) |
| Data Source Year | 1990 U.S. Census |

The dataset contains housing statistics aggregated at the level of census block groups across California. For this task, only `MedInc` was used as the predictor to maintain the simplicity of a single-variable regression and ensure the results are fully visualisable.

---

## 4. Methodology

### 4.1 Data Preparation
The dataset was loaded using scikit-learn and converted to a `pandas DataFrame`. The predictor `MedInc` and target `MedHouseVal` were extracted as NumPy arrays.

### 4.2 Train/Test Split
The data was divided into training (80%) and testing (20%) subsets using `train_test_split` with `random_state=42` to ensure reproducibility.

| Split | Instances |
|---|---|
| Training Set | 16,512 |
| Test Set | 4,128 |

### 4.3 Model Training
A `LinearRegression` model from scikit-learn was instantiated and trained on the training set. scikit-learn's implementation uses the standard OLS closed-form solution.

### 4.4 Prediction and Evaluation
The trained model was used to generate predictions on the test set. Three metrics were computed:

- **Mean Squared Error (MSE):** Average of squared differences between actual and predicted values.
- **Root Mean Squared Error (RMSE):** Square root of MSE — in the same units as the target variable.
- **R-squared (R²):** Proportion of variance in the target variable explained by the model.

### 4.5 Visualisations
Two plots were produced:
1. **Scatter + Regression Line:** Actual data points overlaid with the fitted regression line.
2. **Residual Plot:** Predicted values on the x-axis vs. residuals (actual − predicted) on the y-axis.

---

## 5. Results and Analysis

### 5.1 Fitted Model Parameters

| Parameter | Value |
|---|---|
| Intercept (β₀) | ≈ 0.45 |
| Coefficient (β₁) | ≈ 0.42 |

**Interpretation:** For every one-unit increase in median income, the predicted median house value increases by approximately $42,000.

### 5.2 Performance Metrics

| Metric | Value | Interpretation |
|---|---|---|
| MSE | ≈ 0.72 | Average squared error (in $100k² units) |
| RMSE | ≈ 0.85 | On average, predictions are off by ~$85,000 |
| R² | ≈ 0.47 | Income explains ~47% of variance in house prices |

### 5.3 Regression Plot Analysis
The regression line captures the general upward trend between income and house value. However, the scatter around the line is substantial, particularly at higher income levels, indicating that additional predictors are needed for more accurate predictions.

### 5.4 Residual Plot Analysis
The residual plot reveals a **fan-shaped (heteroscedastic) pattern** — the spread of residuals increases as predicted values increase. This violates the OLS assumption of homoscedasticity (constant error variance). Specifically:
- Low-priced houses are predicted reasonably well.
- High-priced houses are systematically underestimated.

This is partially due to a data artefact: house values are capped at $500,000 ($5.0 in scaled units) in this dataset, causing a cluster of points at the upper boundary that distort the model.

---

## 6. Conclusion

The simple linear regression model demonstrates a statistically meaningful positive relationship between median income and median house value in California. An R² of approximately 0.47 indicates that income is a useful but incomplete predictor — it accounts for roughly half the variation in house prices. The remaining variance is attributable to other factors not included in this model: proximity to the coast, number of rooms, local amenities, and neighbourhood characteristics.

The residual analysis confirms that the linear relationship is an adequate first approximation but that the model's assumptions are violated at higher price ranges. A more robust model would incorporate multiple predictors, non-linear terms, or transformations of the target variable.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task2.ipynb` | Jupyter notebook with all code and inline outputs |
| `task2_regression.png` | Left: scatter + regression line. Right: residual plot |

---

## 8. Tools and Libraries

| Library | Purpose |
|---|---|
| pandas, numpy | Data handling and numerical operations |
| scikit-learn | Dataset loader, train/test split, LinearRegression, metrics |
| matplotlib | Visualisation |
