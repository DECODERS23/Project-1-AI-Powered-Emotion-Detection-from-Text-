"""
STEP 1 - Data Loading & Text Preprocessing
============================================
Downloads the text-emotion dataset via kagglehub and applies:
  - Lowercasing
  - Removing special characters / punctuation / numbers
  - Tokenization (NLTK)
  - Stopword removal (NLTK)
  - Lemmatization (NLTK WordNet)

Output: cleaned_data.csv  (columns: clean_text, emotion)

NOTE ON DATASET:
-----------------
"ananthu017/emotion-detection-fer" (FER-2013) is an IMAGE dataset (faces),
NOT text -- it cannot be used with NLTK / TF-IDF / Logistic Regression.

This script instead uses "praveengovi/emotions-dataset-for-nlp", a Kaggle
text dataset with sentences + emotion labels, which is the correct fit for
this NLP pipeline. If you have a different text dataset, just change
KAGGLE_DATASET below and update TEXT_COLUMN / LABEL_COLUMN to match its
columns.
"""

import os
import re
import pandas as pd
import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# ------------------------------------------------------------------
# CONFIG - adjust if you use a different dataset
# ------------------------------------------------------------------
KAGGLE_DATASET = "praveengovi/emotions-dataset-for-nlp"
TEXT_COLUMN = "text"
LABEL_COLUMN = "emotion"

# Choose "lemmatize" (recommended) or "stem"
NORMALIZATION = "lemmatize"
# ------------------------------------------------------------------

# Download required NLTK resources (only happens once, then cached)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def load_dataset() -> pd.DataFrame:
    """Download dataset with kagglehub and load all text files into one DataFrame."""
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Path to dataset files:", path)

    frames = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)

        if fname.endswith(".txt"):
            # praveengovi dataset format: "sentence;emotion" per line, no header
            df = pd.read_csv(fpath, sep=";", header=None,
                             names=[TEXT_COLUMN, LABEL_COLUMN], engine="python")
            frames.append(df)
        elif fname.endswith(".csv"):
            df = pd.read_csv(fpath)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No .txt/.csv files found in {path}. "
            f"Inspect the folder manually and adjust load_dataset()."
        )

    data = pd.concat(frames, ignore_index=True)
    return data


def clean_text(text: str) -> str:
    """Full preprocessing pipeline: lowercase -> clean -> tokenize -> stopwords -> normalize."""
    # 1. Lowercasing
    text = str(text).lower()

    # 2. Remove special characters, digits, punctuation (keep letters + spaces only)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Stopword removal (also drop single-character tokens)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 5. Stemming / Lemmatization
    if NORMALIZATION == "stem":
        tokens = [stemmer.stem(t) for t in tokens]
    else:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    print("Downloading dataset...")
    df = load_dataset()
    print("Raw data shape:", df.shape)
    print(df.head())

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    print("\nCleaning text (this can take a minute for large datasets)...")
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

    # Drop rows that became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]

    print("\nLabel distribution:")
    print(df[LABEL_COLUMN].value_counts())

    out = df[["clean_text", LABEL_COLUMN]].rename(columns={LABEL_COLUMN: "emotion"})
    out.to_csv("cleaned_data.csv", index=False)
    print("\nSaved cleaned_data.csv  ->", out.shape)
