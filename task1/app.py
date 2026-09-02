import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

st.set_page_config(page_title="Task 1: Iris EDA Explorer", page_icon="🌸", layout="wide")

st.title("🌸 Task 1: Exploratory Data Analysis (Iris Dataset)")
st.markdown("Interactive dashboard exploring feature distributions, summary statistics, and correlations.")

# Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.feature_names, iris.target_names

df, feature_names, target_names = load_data()

# Sidebar
st.sidebar.header("Navigation & Filters")
selected_species = st.sidebar.multiselect(
    "Filter by Species",
    options=list(target_names),
    default=list(target_names)
)

filtered_df = df[df['species'].isin(selected_species)]

# Main Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📈 Feature Distributions", "🔍 Scatter Analysis", "🔥 Correlation Matrix"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(filtered_df))
    col2.metric("Features", len(feature_names))
    col3.metric("Selected Classes", len(selected_species))

    st.subheader("Data Preview")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe().round(3), use_container_width=True)

with tab2:
    st.subheader("Feature Distributions by Species")
    feature_to_plot = st.selectbox("Select Feature for Histogram", feature_names)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    for species in selected_species:
        subset = filtered_df[filtered_df['species'] == species]
        ax.hist(subset[feature_to_plot], bins=15, alpha=0.6, label=str(species))
    ax.set_title(f"Distribution of {feature_to_plot}")
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Interactive 2D Feature Scatter Plot")
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("X-Axis Feature", feature_names, index=2)
    y_axis = col2.selectbox("Y-Axis Feature", feature_names, index=3)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'setosa': '#2b5c8f', 'versicolor': '#e26d5c', 'virginica': '#38b000'}
    for species in selected_species:
        subset = filtered_df[filtered_df['species'] == species]
        ax.scatter(subset[x_axis], subset[y_axis], label=str(species),
                   color=colors.get(species, 'gray'), alpha=0.8, s=60, edgecolors='white')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{x_axis} vs. {y_axis}")
    ax.legend()
    st.pyplot(fig)

with tab4:
    st.subheader("Pearson Correlation Heatmap")
    numeric_df = filtered_df.drop(columns=['species'])
    if len(numeric_df) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, square=True)
        st.pyplot(fig)
    else:
        st.warning("Please select at least one species to compute correlations.")
