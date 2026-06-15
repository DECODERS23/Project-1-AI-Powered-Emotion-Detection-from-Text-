"""
AI-Powered Emotion Detection from Text (Single-File Pipeline)
================================================================
Pipeline:
  1. Download dataset via kagglehub
  2. Text preprocessing with NLTK
       - Lowercasing
       - Removing special characters
       - Tokenization
       - Stopword removal
       - Stemming / Lemmatization
  3. Feature extraction ("input layer")
       - Bag-of-Words (CountVectorizer)
       - TF-IDF (TfidfVectorizer)
  4. Model building - Logistic Regression (trained separately on BoW and TF-IDF)
  5. Evaluation - F1 score (macro/weighted), classification report, confusion matrix
  6. Predict emotion for new custom text

------------------------------------------------------------------
IMPORTANT NOTE ON THE DATASET
------------------------------------------------------------------
"ananthu017/emotion-detection-fer" (FER-2013) is an IMAGE dataset of faces,
NOT text -- it cannot be used with NLTK / BoW / TF-IDF / Logistic Regression.

This script downloads "simaanjali/emotion-analysis-based-on-text", a Kaggle
TEXT dataset (sentence + emotion label), using kagglehub.

KAGGLE_DATASET must be the dataset HANDLE "owner/dataset-name" -- NOT the
local cache path that dataset_download() returns/prints (e.g. do NOT use
"C:/Users/<you>/.cache/kagglehub/datasets/.../versions/1" as the value).
kagglehub caches downloads automatically, so re-running the script reuses
the cached files without re-downloading.

TEXT_COLUMN / LABEL_COLUMN are auto-detected from the dataset's columns
(printed at runtime). If auto-detection fails or picks the wrong columns,
set them manually near the top of the script.
------------------------------------------------------------------

Setup:
  pip install -r requirements.txt   (kagglehub, pandas, numpy, nltk,
                                      scikit-learn, matplotlib, seaborn, joblib)

  kagglehub needs Kaggle API credentials:
    - place kaggle.json in ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json), OR
    - set env vars KAGGLE_USERNAME and KAGGLE_KEY

Run:
  python emotion_detection_nltk.py
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ====================================================================
# CONFIG -- adjust here if you use a different dataset
# ====================================================================
# IMPORTANT: this must be the Kaggle dataset HANDLE "owner/dataset-name",
# NOT a local file path. kagglehub.dataset_download() downloads (or reuses
# its local cache) and RETURNS the local path itself -- you never pass a
# path in here. Passing a path (e.g. "C:\Users\...\versions\1") causes
# "Invalid dataset handle" errors.
KAGGLE_DATASET = "simaanjali/emotion-analysis-based-on-text"

# If left as None, the script will try to auto-detect the text/label
# columns from the candidates below and print what it found. Set these
# manually (e.g. TEXT_COLUMN = "Comment") if auto-detection picks the
# wrong columns or raises an error.
TEXT_COLUMN = None
LABEL_COLUMN = None

TEXT_COLUMN_CANDIDATES = ["text", "Text", "comment", "Comment", "sentence",
                           "Sentence", "content", "Content", "Tweet", "tweet"]
LABEL_COLUMN_CANDIDATES = ["emotion", "Emotion", "label", "Label",
                            "sentiment", "Sentiment", "class", "Class",
                            "Emotions", "emotions"]

NORMALIZATION = "lemmatize"   # "lemmatize" (recommended) or "stem"
MAX_FEATURES = 5000           # vocabulary size for BoW and TF-IDF
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ====================================================================


# --------------------------------------------------------------
# 0. NLTK setup
# --------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# --------------------------------------------------------------
# 1. Load dataset (kagglehub)
# --------------------------------------------------------------
def _find_column(columns, candidates, kind):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(
        f"Could not auto-detect the {kind} column.\n"
        f"Available columns: {list(columns)}\n"
        f"Set TEXT_COLUMN / LABEL_COLUMN manually at the top of the script "
        f"to one of these names."
    )


def load_dataset() -> pd.DataFrame:
    global TEXT_COLUMN, LABEL_COLUMN

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Path to dataset files:", path)
    print("Files found:", os.listdir(path))

    frames = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)

        if fname.endswith(".csv"):
            df = pd.read_csv(fpath)
            print(f"\n{fname} -> columns: {list(df.columns)}, shape: {df.shape}")
            print(df.head(2))
            frames.append(df)
        elif fname.endswith(".txt"):
            # e.g. praveengovi-style "sentence;emotion" lines, no header
            df = pd.read_csv(fpath, sep=";", header=None,
                             names=["text", "emotion"], engine="python")
            print(f"\n{fname} -> columns: {list(df.columns)}, shape: {df.shape}")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No .csv/.txt files found in {path}. "
            f"Inspect the folder manually and adjust load_dataset()."
        )

    data = pd.concat(frames, ignore_index=True)

    # Auto-detect text/label columns if not set manually above
    if TEXT_COLUMN is None:
        TEXT_COLUMN = _find_column(data.columns, TEXT_COLUMN_CANDIDATES, "text")
    if LABEL_COLUMN is None:
        LABEL_COLUMN = _find_column(data.columns, LABEL_COLUMN_CANDIDATES, "label/emotion")

    print(f"\nUsing TEXT_COLUMN = '{TEXT_COLUMN}', LABEL_COLUMN = '{LABEL_COLUMN}'")

    return data


# --------------------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------------------
def clean_text(text: str) -> str:
    # Lowercasing
    text = str(text).lower()

    # Removing special characters / digits / punctuation
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal (also drop single-character tokens)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Stemming / Lemmatization
    if NORMALIZATION == "stem":
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# --------------------------------------------------------------
# 3. Train + evaluate Logistic Regression on a given feature set
# --------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, name: str):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n========== {name} ==========")
    print(f"F1 score (macro):    {f1_macro:.4f}")
    print(f"F1 score (weighted): {f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted emotion")
    plt.ylabel("Actual emotion")
    plt.tight_layout()
    out_png = f"confusion_matrix_{name}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")

    model_path = f"logreg_model_{name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved {model_path}")

    return model, f1_macro, f1_weighted


# --------------------------------------------------------------
# 4. Predict emotion for new custom text
# --------------------------------------------------------------
def predict_emotion(text: str, model, vectorizer, label_encoder) -> str:
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return label_encoder.inverse_transform([pred])[0]


# ================================================================
# MAIN PIPELINE
# ================================================================
if __name__ == "__main__":

    # ---- Step 1: Load + clean data ----
    print("Downloading dataset...")
    df = load_dataset()
    print("Raw data shape:", df.shape)
    print(df.head())

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    print("\nCleaning text (lowercasing, removing special chars, tokenizing,")
    print("removing stopwords, lemmatizing/stemming)...")
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    print("\nLabel distribution:")
    print(df[LABEL_COLUMN].value_counts())

    # ---- Step 2: Encode labels + train/test split (on raw text) ----
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COLUMN])
    class_names = label_encoder.classes_
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("\nClasses:", list(class_names))

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---- Step 3: Feature extraction (input layer) ----
    # Bag-of-Words
    bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    X_train_bow = bow_vectorizer.fit_transform(X_train_text)
    X_test_bow = bow_vectorizer.transform(X_test_text)
    joblib.dump(bow_vectorizer, "bow_vectorizer.pkl")
    print("\nBag-of-Words input shape:", X_train_bow.shape)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    print("TF-IDF input shape:", X_train_tfidf.shape)

    # ---- Step 4 & 5: Train Logistic Regression + evaluate (F1, confusion matrix) ----
    bow_model, bow_f1_macro, bow_f1_weighted = train_and_evaluate(
        X_train_bow, X_test_bow, y_train, y_test, class_names, "BoW"
    )
    tfidf_model, tfidf_f1_macro, tfidf_f1_weighted = train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test, class_names, "TFIDF"
    )

    print("\n========== Summary ==========")
    print(f"BoW   -> F1 macro: {bow_f1_macro:.4f} | F1 weighted: {bow_f1_weighted:.4f}")
    print(f"TFIDF -> F1 macro: {tfidf_f1_macro:.4f} | F1 weighted: {tfidf_f1_weighted:.4f}")

    # ---- Step 6: Try it on new custom sentences (using TF-IDF model) ----
    print("\n========== Sample Predictions (TF-IDF model) ==========")
    samples = [
        "I can't believe I won the lottery, this is amazing!",
        "I'm so scared of what might happen tomorrow.",
        "This traffic is making me so angry right now.",
        "I miss my old friends so much, feeling really down.",
    ]
    for s in samples:
        emotion = predict_emotion(s, tfidf_model, tfidf_vectorizer, label_encoder)
        print(f"{s!r:60s} -> {emotion}")
