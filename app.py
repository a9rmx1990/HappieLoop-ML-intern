import streamlit as st

st.set_page_config(
    page_title="Machine Learning Internship Portfolio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🚀 ML Portfolio Explorer")
st.sidebar.markdown("Select an interactive project module:")

project = st.sidebar.radio(
    "Choose Project:",
    [
        "🏠 Portfolio Overview",
        "🌸 Task 1: Exploratory Data Analysis",
        "📰 Task 6: NLP Text Classification",
        "🧠 Task 8: Neural Network (PyTorch)",
        "📈 Task 9: Time Series Forecasting (SARIMA)",
        "🔬 Task 10: Clinical Diagnostic SVM"
    ]
)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Each project folder also contains an independent `app.py` that can be run standalone.")

if project == "🏠 Portfolio Overview":
    st.title("🌟 Machine Learning Engineering Internship Portfolio")
    st.markdown("""
    Welcome to the interactive project portal. This submission demonstrates a well-rounded foundation across five key disciplines in Machine Learning, Deep Learning, and NLP.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌸 Task 1: Exploratory Data Analysis (Iris)
        - **Domain:** Data Analysis & Statistical Exploration
        - **Highlights:** Multi-feature distributions, correlation matrices, separability analysis.
        - **Dataset:** Fisher's 1936 Iris Dataset.
        """)

        st.markdown("""
        ### 📰 Task 6: NLP Text Classification (Naive Bayes)
        - **Domain:** Natural Language Processing (NLP)
        - **Highlights:** Live paragraph topic classifier, TF-IDF vectorization, vocabulary log-probability analyzer, **~92% test accuracy**.
        - **Dataset:** 20 Newsgroups.
        """)
        
        st.markdown("""
        ### 🧠 Task 8: Deep Learning (PyTorch from Scratch)
        - **Domain:** Deep Learning & Computer Vision
        - **Highlights:** Custom 3-layer PyTorch MLP with zero-blur center-of-mass image upload, StepLR scheduler, Dropout regularization, **~98% test accuracy**.
        - **Dataset:** MNIST Handwritten Digits.
        """)

    with col2:
        st.markdown("""
        ### 📈 Task 9: Time Series Demand Forecasting
        - **Domain:** Statistical Time Series Modeling
        - **Highlights:** Box-Jenkins methodology, Augmented Dickey-Fuller stationarity testing, SARIMA `(1,1,1)x(1,1,1)[12]`, 24-month horizon forecasting (**3.8% MAPE**).
        - **Dataset:** Monthly Airline Passengers (1949–1960).
        """)

        st.markdown("""
        ### 🔬 Task 10: Clinical Diagnostic SVM Classifier
        - **Domain:** Biomedical Machine Learning & Optimization
        - **Highlights:** 5-Fold cross-validation across 4 kernels, `GridSearchCV` hyperparameter tuning, **ROC-AUC of 0.996**, cytopathology feature ranking.
        - **Dataset:** Breast Cancer Wisconsin (Diagnostic).
        """)

elif project == "🌸 Task 1: Exploratory Data Analysis":
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("task1_app", os.path.abspath("task1/app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

elif project == "📰 Task 6: NLP Text Classification":
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("task6_app", os.path.abspath("task6/app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

elif project == "🧠 Task 8: Neural Network (PyTorch)":
    import importlib.util
    import os
    os.chdir(os.path.abspath("task8"))
    spec = importlib.util.spec_from_file_location("task8_app", os.path.abspath("app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.chdir("..")

elif project == "📈 Task 9: Time Series Forecasting (SARIMA)":
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("task9_app", os.path.abspath("task9/app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

elif project == "🔬 Task 10: Clinical Diagnostic SVM":
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("task10_app", os.path.abspath("task10/app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
