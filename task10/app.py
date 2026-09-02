import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

st.set_page_config(page_title="Task 10: SVM Breast Cancer Diagnosis", page_icon="🔬", layout="wide")

st.title("🔬 Task 10: Breast Cancer Diagnostic SVM Classifier")
st.markdown("Interactive Clinical Decision Support System trained on 569 FNA biopsies with **ROC-AUC of 0.996**.")

@st.cache_data
def load_cancer_data():
    cancer = load_breast_cancer()
    return cancer.data, cancer.target, cancer.feature_names, cancer.target_names

X, y, feature_names, target_names = load_cancer_data()

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Sidebar
st.sidebar.header("SVM Model Settings")
kernel_choice = st.sidebar.selectbox("Select SVM Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
c_val = st.sidebar.slider("Regularization (C)", min_value=0.1, max_value=20.0, value=10.0, step=0.5)

@st.cache_resource
def train_svm(kernel, c):
    clf = SVC(kernel=kernel, C=c, probability=True, random_state=42)
    clf.fit(X_train_sc, y_train)
    return clf

model = train_svm(kernel_choice, c_val)

# Predictions
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Interactive Patient Diagnosis", "📊 Model Evaluation & ROC", "🔍 Feature Impact"])

with tab1:
    st.subheader("Simulate Clinical Biopsy Measurements")
    st.markdown("Adjust key nuclear morphological features below to evaluate tumor malignancy prediction:")

    col1, col2, col3 = st.columns(3)
    
    # Feature inputs using medians from test dataset as default
    f_mean_radius = col1.slider("Mean Radius", float(X[:, 0].min()), float(X[:, 0].max()), float(np.median(X[:, 0])))
    f_mean_texture = col2.slider("Mean Texture", float(X[:, 1].min()), float(X[:, 1].max()), float(np.median(X[:, 1])))
    f_mean_concavity = col3.slider("Mean Concavity", float(X[:, 6].min()), float(X[:, 6].max()), float(np.median(X[:, 6])))

    # Create dummy vector from mean values
    patient_vector = np.median(X, axis=0)
    patient_vector[0] = f_mean_radius
    patient_vector[1] = f_mean_texture
    patient_vector[6] = f_mean_concavity

    patient_scaled = scaler.transform(patient_vector.reshape(1, -1))
    patient_prob_benign = model.predict_proba(patient_scaled)[0, 1]
    patient_prob_malignant = 1.0 - patient_prob_benign

    st.divider()
    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        if patient_prob_malignant > 0.5:
            st.error(f"### ⚠️ Predicted Diagnosis: **MALIGNANT**")
            st.markdown(f"**Malignancy Probability:** `{patient_prob_malignant * 100:.2f}%`")
        else:
            st.success(f"### ✅ Predicted Diagnosis: **BENIGN**")
            st.markdown(f"**Benign Probability:** `{patient_prob_benign * 100:.2f}%`")

    with res_col2:
        st.markdown("**Probability Gauge**")
        st.progress(float(patient_prob_malignant), text=f"Malignancy Risk Index: {patient_prob_malignant*100:.1f}%")

with tab2:
    st.subheader(f"Performance on Unseen Test Cohort (N = {len(y_test)})")
    
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=target_names, yticklabels=target_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Test Confusion Matrix")
        st.pyplot(fig)
        
    with mcol2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.3f})", color="#2b5c8f", lw=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.subheader("Top Predictive Features (Linear SVM Weights)")
    linear_svm = SVC(kernel='linear', C=1, random_state=42).fit(X_train_sc, y_train)
    coef_series = pd.Series(np.abs(linear_svm.coef_[0]), index=feature_names).sort_values().tail(10)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    coef_series.plot(kind='barh', color='#e26d5c', ax=ax)
    ax.set_title("10 Most Influential FNA Nuclear Characteristics")
    ax.set_xlabel("Absolute Weight Magnitude")
    st.pyplot(fig)
