# HappieLoop Machine Learning Internship (HappieLoop-ML-intern)

Welcome to the **HappieLoop Machine Learning Internship** repository! This repository contains a comprehensive suite of Machine Learning & Deep Learning tasks, interactive Streamlit web applications, Jupyter analysis notebooks, and technical documentation.

---

## 🚀 Overview of Tasks

| Task | Title | Focus Area / Key Algorithms | Output & Features |
|---|---|---|---|
| **Task 1** | Exploratory Data Analysis | Data Preprocessing, Feature Distributions, Correlation Heatmaps | Interactive EDA Dashboard & Visualizations |
| **Task 2** | Linear Regression & Feature Engineering | Scikit-Learn Regression, Polynomial Features, Metrics ($R^2$, RMSE) | Regression Curve Plots & Metric Evaluations |
| **Task 3** | Pretrained Image Classification | Transfer Learning (MobileNetV2, ResNet50, PyTorch / TensorFlow) | Image Classifier with Top-5 Probabilities |
| **Task 4** | Customer Segmentation | K-Means Clustering, Elbow Method, Silhouette Score | 2D/3D Cluster Maps & Segment Profiling |
| **Task 5** | Decision Trees & Random Forests | DecisionTree, RandomForest, Feature Importance, Depth Sweep | Confusion Matrices & Tree Visualizations |
| **Task 6** | Text Classification & NLP | TF-IDF Vectorization, Multinomial Naive Bayes, SGDClassifier | Interactive Text Classifier Web App |
| **Task 7** | Dimensionality Reduction | Principal Component Analysis (PCA), Variance Ratio Analysis | 2D/3D Interactive Projection Visualizer |
| **Task 8** | Deep Learning Image Classifier | PyTorch Convolutional Neural Networks (CNN), Training Curves | Trained PyTorch Model (`mnist_model.pth`) |
| **Task 9** | Time Series Forecasting | Statsmodels ARIMA / SARIMAX, ACF/PACF Diagnostics | Multi-step Ahead Forecast Plots & App |
| **Task 10** | Support Vector Machines (SVM) | Linear, RBF, Polynomial Kernels, Hyperparameter Tuning | Kernel Comparison & Decision Boundary Visualizer |

---

## 🛠️ Project Structure

```
HappieLoop-ML-intern/
├── app.py                   # Master Multi-Page Streamlit Dashboard
├── requirements.txt         # Project Dependencies
├── task1/                   # Task 1: Exploratory Data Analysis
│   ├── app.py
│   ├── task1.ipynb
│   └── README.md
├── task2/                   # Task 2: Linear Regression & Feature Engineering
│   ├── task2.ipynb
│   └── README.md
├── task3/                   # Task 3: Pretrained Image Classification
│   ├── task3.ipynb
│   └── README.md
├── task4/                   # Task 4: Customer Segmentation (K-Means)
│   ├── task4.ipynb
│   └── README.md
├── task5/                   # Task 5: Decision Trees & Random Forests
│   ├── task5.ipynb
│   └── README.md
├── task6/                   # Task 6: NLP & Text Classification
│   ├── app.py
│   ├── task6.ipynb
│   └── README.md
├── task7/                   # Task 7: Dimensionality Reduction (PCA)
│   ├── task7.ipynb
│   └── README.md
├── task8/                   # Task 8: PyTorch CNN Classifier
│   ├── app.py
│   ├── task8.ipynb
│   ├── mnist_model.pth
│   └── README.md
├── task9/                   # Task 9: Time Series Forecasting (ARIMA)
│   ├── app.py
│   ├── task9.ipynb
│   └── README.md
├── task10/                  # Task 10: Support Vector Machines (SVM)
│   ├── app.py
│   ├── task10.ipynb
│   └── README.md
└── README.md
```

---

## 💻 Quick Start & Setup

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone git@github.com:a9rmx1990/HappieLoop-ML-intern.git
cd HappieLoop-ML-intern

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Interactive Streamlit Dashboard

Run the main application to explore all tasks from a unified web interface:

```bash
streamlit run app.py
```

Or run individual task web apps:

```bash
# Task 6 (NLP Classifier):
streamlit run task6/app.py

# Task 8 (PyTorch Neural Net Digit Recognizer):
streamlit run task8/app.py

# Task 9 (Time Series Forecast):
streamlit run task9/app.py

# Task 10 (SVM Kernel Playground):
streamlit run task10/app.py
```

---

## 📊 Summary of Tech Stack

- **Core Languages & Tools**: Python, Jupyter Notebooks, Git
- **Data & Numerical Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-Learn, Statsmodels
- **Deep Learning Frameworks**: PyTorch, torchvision
- **Web App Framework**: Streamlit
- **Data Visualization**: Matplotlib, Seaborn, Plotly

---

## 📜 License & Acknowledgments
Developed as part of the HappieLoop Machine Learning Internship program.
