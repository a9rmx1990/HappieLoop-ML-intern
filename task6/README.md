---
**Internship Task Report**
**Task No.:** 6
**Task Title:** Text Classification with Naive Bayes
**Date:** August 2026
**Tool / Notebook:** `task6.ipynb`

---

## 1. Objective

The objective of this task was to build a text classification pipeline using the Multinomial Naive Bayes algorithm with TF-IDF feature extraction, evaluate its performance on a multi-class newsgroup classification problem, and analyse the most predictive vocabulary per category.

---

## 2. Introduction

Text classification is one of the most impactful applications of machine learning, underlying spam filters, content moderation, news categorisation, sentiment analysis, and customer support routing. The challenge lies in converting unstructured text — sequences of words — into numerical representations that a machine learning model can process.

The **Bag-of-Words (BoW)** model is the simplest such representation: it counts word occurrences per document, ignoring order. **TF-IDF** (Term Frequency-Inverse Document Frequency) improves on this by down-weighting words that appear frequently across all documents (such as "the", "is") and up-weighting words that are distinctive to specific documents.

**Multinomial Naive Bayes** is a probabilistic classifier that applies Bayes' theorem with the "naive" assumption that features (words) are conditionally independent given the class label. Despite this simplification, it performs remarkably well on text classification tasks and remains a strong baseline against which more complex models are measured.

This task used a subset of the **20 Newsgroups dataset** — a collection of approximately 20,000 newsgroup posts across 20 topics — focusing on four semantically distinct categories.

---

## 3. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | 20 Newsgroups |
| Source | `sklearn.datasets.fetch_20newsgroups()` |
| Total Available Categories | 20 |
| Categories Used | sci.space, rec.sport.baseball, talk.politics.guns, comp.graphics |
| Train Samples | ~2,200 posts |
| Test Samples | ~1,500 posts |
| Preprocessing Applied | Removed email headers, footers, and quoted replies |
| Storage | Local `data_20news/` directory |

Email headers and footers were explicitly removed using sklearn's `remove` parameter to prevent the model from learning metadata artefacts (e.g., author email addresses or posting signatures) rather than actual content.

---

## 4. Methodology

### 4.1 Text Preprocessing
A lightweight cleaning function was applied to each document:
- Convert all text to lowercase
- Remove numeric digits
- Remove punctuation characters
- Collapse multiple whitespace characters into single spaces

This normalisation reduces vocabulary size and removes noise without losing semantic content.

### 4.2 TF-IDF Vectorisation
Documents were converted into numerical feature vectors using `TfidfVectorizer` with the following configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 15,000 | Limits vocabulary to the 15,000 most common terms |
| `min_df` | 2 | Excludes terms appearing in fewer than 2 documents |
| `stop_words` | 'english' | Removes common English function words |

The resulting feature matrix is **sparse** — most entries are zero since each document contains only a small fraction of the total vocabulary.

The vectoriser was **fit only on the training set** and applied as a transform to the test set. This is critical to prevent data leakage — the test set vocabulary statistics must not influence the feature representation.

### 4.3 Model Training
A `MultinomialNB` classifier was trained with Laplace smoothing (`alpha=0.1`). Laplace smoothing prevents zero-probability estimates for words that appear in the test set but not in any training document for a given class.

### 4.4 Evaluation
Performance was assessed using:
- **Accuracy:** Overall correct classification rate
- **Precision, Recall, F1-score:** Per-class metrics
- **Confusion matrix:** Visual analysis of inter-class confusion
- **Top-10 log-probability words per class:** Analysis of the most discriminative vocabulary

---

## 5. Results and Analysis

### 5.1 Classification Performance

**Overall Accuracy: ~90–92%**

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| comp.graphics | ~0.93 | ~0.91 | ~0.92 |
| rec.sport.baseball | ~0.97 | ~0.96 | ~0.97 |
| sci.space | ~0.96 | ~0.95 | ~0.96 |
| talk.politics.guns | ~0.85 | ~0.88 | ~0.87 |

`rec.sport.baseball` and `sci.space` achieved the highest F1-scores due to their domain-specific and relatively unique vocabularies. `talk.politics.guns` showed the lowest scores — political discourse tends to employ general vocabulary (government, people, law, right) that overlaps with other categories.

### 5.2 Confusion Matrix Analysis
The confusion matrix revealed minimal cross-category confusion. The few misclassifications that did occur were concentrated between `sci.space` and `comp.graphics`, likely due to shared technology-related terms, and within `talk.politics.guns` which occasionally borrowed vocabulary from other topic areas.

### 5.3 Most Predictive Vocabulary

The top-10 log-probability words per category (extracted from `nb.feature_log_prob_`) reveal that the model learned semantically meaningful representations:

| Category | Top Discriminative Terms |
|---|---|
| sci.space | space, nasa, earth, orbit, launch, shuttle, mission, moon, solar |
| rec.sport.baseball | game, year, player, hit, run, team, season, pitching, baseball |
| talk.politics.guns | gun, people, government, law, right, weapons, firearms, state |
| comp.graphics | image, file, format, color, program, software, window, display |

These terms are intuitively aligned with the respective domains, confirming that the model has genuinely learned content-based discrimination rather than superficial patterns.

---

## 6. Technical Note: Data Download Issue

During initial notebook execution, an `OSError: Directory not empty` was encountered when sklearn attempted to extract the downloaded dataset archive into a pre-existing directory from a previous partial run. This was resolved by:
1. Redirecting downloads to a local `data_20news/` directory using the `data_home` parameter
2. Programmatically clearing any existing partially extracted data before re-downloading

This approach ensures clean, reproducible execution across repeated notebook runs.

---

## 7. Conclusion

The Multinomial Naive Bayes classifier combined with TF-IDF feature extraction achieved approximately 90–92% accuracy on a four-class newsgroup classification task. The model generalises well despite the strong independence assumption and provides interpretable evidence of what it has learned through the per-class vocabulary analysis. The lowest performance was observed on the `talk.politics.guns` category, attributable to its generic vocabulary. This task demonstrates that classic NLP methods remain competitive baselines and are particularly valuable when interpretability and training speed are priorities.

---

## 8. Output Artifacts

| File | Description |
|---|---|
| `task6.ipynb` | Jupyter notebook with all code and inline outputs |
| `task6_confusion.png` | Confusion matrix across 4 categories |
| `task6_top_words.png` | Top-10 predictive words per category |
| `data_20news/` | Downloaded 20 Newsgroups dataset |

---

## 9. Tools and Libraries

| Library | Purpose |
|---|---|
| scikit-learn | fetch_20newsgroups, TfidfVectorizer, MultinomialNB, metrics |
| seaborn | Confusion matrix heatmap |
| matplotlib | Visualisation |
| re, string | Text preprocessing |
