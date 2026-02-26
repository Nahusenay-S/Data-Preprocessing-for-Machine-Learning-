# 📊 Social Media Sentiment Analysis: Data Preprocessing

This repository contains a professional-grade preprocessing pipeline designed for **Sentiment Analysis**. It transforms messy, raw social media data into a high-quality feature set ready for training Machine Learning models.

---

## 🚀 Overview

The original dataset contained **732 records** with high-cardinality labels (**279 unique sentiments**) and raw text content. This project provides a robust solution to:
1.  **Consolidate Labels**: Group 279 chaotic sentiment strings into **4 clean core classes**.
2.  **Extract NLP Signals**: Convert raw text into numerical features using **TF-IDF Vectorization**.
3.  **Standardize Metadata**: Normalize engagement metrics (likes, retweets) and temporal features.

## 🛠️ Tech Stack

- **Python 3.14**
- **Pandas**: Data manipulation and cleaning.
- **Scikit-Learn**: Vectorization, scaling, label encoding, and dataset splitting.
- **NumPy**: Numerical operations.

---

## 🔄 Preprocessing Pipeline

The `enhanced_preprocess.py` script executes the following steps:

### 1. Data Unification
- Strips inconsistent whitespace from columns like `Platform` and `Country`.
- Merges near-duplicate categories (e.g., `'Twitter '` and `'Twitter'`).

### 2. Sentiment Mapping (Target Engineering)
Maps 279 strings into a unified target space:
- **Positive**: Joy, Happiness, Excitement, Gratitude, etc.
- **Negative**: Anger, Sadness, Fear, Despair, etc.
- **Neutral**: Calmness, Curiosity, Acceptance, etc.
- **Other**: For labels that do not fit the core categories.

### 3. NLP & Feature Engineering
- **TF-IDF Vectorization**: Extracts the **top 100 most significant words** from the post content.
- **Hashtag Count**: Derives the number of tags as a numerical feature.
- **Standard Scaling**: Normalizes `Retweets`, `Likes`, and temporal data (`Year`, `Month`, `Day`, `Hour`) to a mean of 0 and variance of 1.

### 4. Training/Test Split
- Performs an **80/20 stratified-style split** for reliable model evaluation.

---

## 📥 Getting Started

### Prerequisites
Install the required libraries:
```bash
pip install pandas numpy scikit-learn
```

### Running the Pipeline
Simply execute the enhanced script to generate the training and testing datasets:
```bash
python enhanced_preprocess.py
```

---

## 📂 Output Files

The script generates versioned CSV files for your model:
- `X_train_v2.csv`: Training features (109 columns).
- `y_train_v2.csv`: Training target labels.
- `X_test_v2.csv`: Testing features.
- `y_test_v2.csv`: Testing target labels.

---

## 📝 License

This project is open-source and available under the MIT License.
