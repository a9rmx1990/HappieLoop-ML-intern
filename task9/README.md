---
**Internship Task Report**
**Task No.:** 9
**Task Title:** Time Series Forecasting with ARIMA / SARIMA
**Date:** August 2026
**Tool / Notebook:** `task9.ipynb`

---

## 1. Objective

The objective of this task was to perform end-to-end time series analysis and forecasting on sequential data using the ARIMA / SARIMA methodology. The task covers exploratory data analysis of time-dependent patterns, statistical stationarity testing (Augmented Dickey-Fuller test), ACF and PACF diagnostic analysis, model parameter identification, model fitting, and multi-step out-of-sample forecasting with performance evaluation.

---

## 2. Introduction

Time series forecasting is a specialized domain of machine learning and statistical modeling where observations are ordered chronologically and exhibit temporal dependencies. Unlike standard regression where samples are assumed to be independent and identically distributed (i.i.d.), time series data inherently involves autocorrelation, trends, seasonality, and cyclic fluctuations.

The **ARIMA (AutoRegressive Integrated Moving Average)** framework is the cornerstone of statistical time series modeling:
- **AR (AutoRegressive - p):** Models the dependency between an observation and a specified number of lagged observations.
- **I (Integrated - d):** Differencing applied to non-stationary data to make it stationary (constant mean and variance over time).
- **MA (Moving Average - q):** Models the dependency between an observation and a residual error from a moving average model applied to lagged observations.

When data exhibits periodic seasonal behavior (e.g., annual cycles), the model is extended to **SARIMA (Seasonal ARIMA)** denoted as `SARIMA(p, d, q) × (P, D, Q)[s]`, where `s` is the seasonal periodicity and `P, D, Q` are seasonal autoregressive, differencing, and moving average components respectively.

In this task, SARIMA was applied to the classic **Monthly Airline Passengers Dataset** (1949–1960) to forecast passenger volumes 24 months into the future.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | Monthly Airline Passengers (Box-Jenkins Airline Dataset) |
| Source | `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv` |
| Time Horizon | January 1949 – December 1960 (12 years) |
| Frequency | Monthly (`MS` - Month Start) |
| Total Observations | 144 monthly recordings |
| Target Variable | Number of passengers (in thousands) |
| Train / Test Split | First 120 months (1949–1958) training / Last 24 months (1959–1960) test |

---

## 4. Methodology

### 4.1 Data Exploration & Transformation
Initial visualization of the raw series revealed two prominent characteristics:
1. **Upward Trend:** Overall passenger count steadily increases over the 12-year timeline.
2. **Multiplicative Seasonality:** Annual seasonal peaks occur consistently every summer, and the amplitude of seasonal fluctuations expands as the trend rises.

To stabilize the expanding variance (convert multiplicative seasonality into additive seasonality), a **natural log transformation** was applied:
```python
df_log = np.log(df)
```

### 4.2 Stationarity Testing (Augmented Dickey-Fuller Test)
ARIMA models require the underlying data to be stationary (constant mean, constant variance, and autocovariance independent of time). The **Augmented Dickey-Fuller (ADF)** test was conducted:
- **Null Hypothesis ($H_0$):** The series has a unit root (non-stationary).
- **Alternative Hypothesis ($H_1$):** The series is stationary ($p < 0.05$).

Transformations applied to achieve stationarity:
1. First-order differencing ($d = 1$) to remove the linear trend.
2. Seasonal differencing at lag 12 ($D = 1, s = 12$) to eliminate the 12-month annual cycle.

```python
df_log_diff = df_log.diff(12).diff(1)
```

### 4.3 Diagnostic Plots (ACF and PACF)
- **Autocorrelation Function (ACF):** Measures correlation between series values separated by $k$ time intervals. Helps identify moving average order $q$ and seasonal $Q$.
- **Partial Autocorrelation Function (PACF):** Measures correlation between series values separated by $k$ intervals after controlling for intermediate values. Helps identify autoregressive order $p$ and seasonal $P$.

Plots generated using `statsmodels.graphics.tsaplots` guided the parameter selection.

### 4.4 Model Fitting & Selection
A Seasonal ARIMA model was fitted to the log-transformed training set:
```python
sarima = SARIMAX(
    train_log,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
result = sarima.fit(disp=False)
```

### 4.5 Multi-step Forecasting and Evaluation
A 24-step out-of-sample forecast was generated with 95% confidence intervals. Point forecasts and confidence boundaries were transformed back from log space to original passenger counts using $\exp(\cdot)$.

Forecast accuracy was evaluated against actual test data using:
- **Root Mean Squared Error (RMSE):** Average error magnitude in passenger units (thousands).
- **Mean Absolute Percentage Error (MAPE):** Relative percentage error across test timestamps.

---

## 5. Results and Analysis

### 5.1 Stationarity Test Results

| Series Version | ADF Statistic | p-value | Conclusion |
|---|---|---|---|
| Original Raw Series | +0.815 | 0.9918 | Highly Non-Stationary |
| Log-Transformed Series | -1.717 | 0.4223 | Non-Stationary |
| **Log + Seasonal diff(12) + First diff(1)** | **-3.125** | **0.0248** | **Stationary ($p < 0.05$)** |

The combination of log transformation, seasonal differencing ($D=1$), and first differencing ($d=1$) successfully brought the series below the 5% significance threshold, validating the integration parameters for SARIMA.

### 5.2 Model Estimation

The maximum likelihood estimation converged efficiently:
- Model: `SARIMAX(1, 1, 1) × (1, 1, 1, 12)`
- Log-Likelihood: Substantially optimized compared to non-seasonal baselines.
- Residual analysis verified that model residuals behaved largely as uncorrelated Gaussian white noise.

### 5.3 Forecasting Performance (24-Month Horizon)

| Metric | Value | Interpretation |
|---|---|---|
| **RMSE** | **~21.5k passengers** | Average prediction deviation over a 2-year forecast horizon |
| **MAPE** | **~3.8%** | Outstanding relative precision (< 5% error) |

### 5.4 Visualization and Forecast Dynamics
The generated plot (`task9_forecast.png`) highlights:
- The SARIMA forecast precisely captured both the continuing long-term upward trend and the steep annual summer peaks.
- The true 1959–1960 test observations remained entirely within the model's shaded 95% confidence interval.
- Confidence intervals progressively expand over time, naturally reflecting increasing uncertainty further out into the forecast horizon.

---

## 6. Key Insights and Takeaways

- **Log-transformation is critical for heteroscedastic time series:** When seasonal amplitude scales with the level of the series, linear additive models fail unless the series is converted to log space.
- **Seasonal differencing ($D=1, s=12$) is mandatory for periodic cycles:** Standard single differencing ($d=1$) only removes trend; it leaves significant annual autocorrelation intact.
- **Confidence intervals convey risk:** In business and operational forecasting, point predictions are insufficient; confidence intervals allow capacity planners to prepare for best- and worst-case demand scenarios.

---

## 7. Output Artifacts

| File | Description |
|---|---|
| `task9.ipynb` | Jupyter notebook containing full analysis, modeling, and output cells |
| `task9_raw.png` | Time series visualization: raw passenger counts vs log-transformed series |
| `task9_acf_pacf.png` | Autocorrelation (ACF) and Partial Autocorrelation (PACF) diagnostic plots |
| `task9_forecast.png` | 24-month SARIMA forecast vs actual observations with 95% confidence bands |

---

## 8. Tools and Libraries

| Library | Purpose |
|---|---|
| statsmodels | `SARIMAX`, `adfuller`, `plot_acf`, `plot_pacf` |
| pandas | Datetime indexing (`DatetimeIndex`), date slicing, frequency handling |
| numpy | Numerical transformations (`log`, `exp`) and error metrics |
| scikit-learn | `mean_squared_error` |
| matplotlib | Time series and diagnostic visualization |
