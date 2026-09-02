import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os
import shutil

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Task 6: NLP Text Classifier", page_icon="📰", layout="wide")

st.title("📰 Task 6: Real-Time NLP Topic Classification (Naive Bayes & TF-IDF)")
st.markdown("Classify news articles, emails, or custom text into topics using **TF-IDF Vectorization** and **Multinomial Naive Bayes** (~92% test accuracy).")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

CATEGORIES = [
    'sci.space',
    'rec.sport.baseball',
    'talk.politics.guns',
    'comp.graphics'
]

DISPLAY_NAMES = {
    'sci.space': 'Space & Astronomy',
    'rec.sport.baseball': 'Baseball & Sports',
    'talk.politics.guns': 'Politics & Firearms',
    'comp.graphics': 'Computer Graphics'
}

# Resolve local data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_HOME = os.path.join(BASE_DIR, 'data_20news')

@st.cache_resource
def train_nlp_pipeline():
    train_data = fetch_20newsgroups(
        subset='train',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        data_home=DATA_HOME
    )
    test_data = fetch_20newsgroups(
        subset='test',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        data_home=DATA_HOME
    )
    
    X_train_cleaned = [clean_text(t) for t in train_data.data]
    X_test_cleaned  = [clean_text(t) for t in test_data.data]
    
    vectorizer = TfidfVectorizer(max_features=15000, min_df=2, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_cleaned)
    X_test_tfidf = vectorizer.transform(X_test_cleaned)
    
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, train_data.target)
    
    return nb, vectorizer, train_data.target_names, test_data, X_test_tfidf

with st.spinner("Loading TF-IDF + Naive Bayes NLP model..."):
    model, vectorizer, target_names, test_data, X_test_tfidf = train_nlp_pipeline()

# Preset examples
PRESET_SAMPLES = {
    "Select an example...": "",
    "Space Exploration": "NASA successfully launched the new telescope into orbit around the sun to study distant galaxies and cosmic radiation.",
    "Baseball Match": "The starting pitcher struck out ten batters in the ninth inning to secure the team's victory in the championship series.",
    "Computer Graphics": "We rendered the 3D polygon mesh using OpenGL shaders with realistic ray tracing and texture mapping.",
    "Gun Legislation": "The Supreme Court discussed the second amendment rights and firearm regulations passed by the state legislature."
}

# Tabs
tab1, tab2, tab3 = st.tabs(["✍️ Live Text Classifier", "📊 Category Vocabulary Explorer", "📈 Model Confusion Matrix"])

with tab1:
    st.subheader("Test the Classifier with Custom Text")
    
    col_sel, col_btn = st.columns([3, 1])
    preset_choice = col_sel.selectbox("Try a sample prompt or type your own below:", list(PRESET_SAMPLES.keys()))
    default_text = PRESET_SAMPLES[preset_choice] if preset_choice != "Select an example..." else ""

    user_input = st.text_area(
        "Enter paragraph or sentence to classify:",
        value=default_text,
        height=140,
        placeholder="e.g. The satellite entered lunar orbit after a successful propulsion burn..."
    )

    if user_input.strip():
        cleaned_input = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])
        
        # Predict
        probs = model.predict_proba(input_tfidf)[0]
        pred_idx = np.argmax(probs)
        pred_category = target_names[pred_idx]
        confidence = probs[pred_idx]
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"### Predicted Topic: **{DISPLAY_NAMES.get(pred_category, pred_category)}**")
            st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            st.progress(float(confidence))
            
            # Words found in vocabulary
            words_in_input = set(cleaned_input.split())
            vocab = set(vectorizer.get_feature_names_out())
            matched_keywords = words_in_input.intersection(vocab)
            if matched_keywords:
                st.markdown(f"**Informative Keywords Detected:**")
                st.write(", ".join([f"`{w}`" for w in sorted(list(matched_keywords))[:12]]))
                
        with res_col2:
            st.subheader("Topic Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            labels = [DISPLAY_NAMES.get(c, c) for c in target_names]
            colors = ['#38b000' if i == pred_idx else '#4361ee' for i in range(len(labels))]
            bars = ax.barh(labels, probs * 100, color=colors)
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 105)
            for bar in bars:
                w = bar.get_width()
                if w > 1:
                    ax.annotate(f'{w:.1f}%',
                                xy=(w, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0), textcoords="offset points",
                                ha='left', va='center', fontsize=9)
            st.pyplot(fig)
    else:
        st.info("💡 Type any text above or choose a preset sample to see real-time classification.")

with tab2:
    st.subheader("Top Predictive Terms Learned per Topic")
    st.markdown("These words carry the highest log-probabilities for each respective category:")
    
    feature_names_arr = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    
    for ax, i, cat in zip(axes.flatten(), range(4), target_names):
        top_idx = np.argsort(model.feature_log_prob_[i])[-10:][::-1]
        top_words = feature_names_arr[top_idx]
        top_scores = np.exp(model.feature_log_prob_[i][top_idx]) * 1000  # relative scale
        
        ax.barh(range(10), top_scores[::-1], color='#2b5c8f', alpha=0.85)
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_words[::-1], fontsize=9)
        ax.set_title(DISPLAY_NAMES.get(cat, cat), fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative Term Weight")
        
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Test Set Confusion Matrix")
    y_test_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(test_data.target, y_test_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    short_labels = ['Graphics', 'Baseball', 'Space', 'Guns']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=short_labels, yticklabels=short_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test Confusion Matrix (N = 1,500)")
    st.pyplot(fig)
