"""
STEP 2 - Feature Extraction (the "input layer" for the model)
================================================================
Reads cleaned_data.csv (produced by 1_data_preprocessing.py) and builds
two separate feature representations:

  - Bag-of-Words (CountVectorizer)
  - TF-IDF       (TfidfVectorizer)

Each is split into train/test sets and saved to disk so Step 3 can train
the model purely from these saved feature matrices, without redoing any
text processing.

Outputs:
  label_encoder.pkl
  bow_vectorizer.pkl   , bow_features.pkl    -> (X_train, X_test, y_train, y_test)
  tfidf_vectorizer.pkl , tfidf_features.pkl  -> (X_train, X_test, y_train, y_test)
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
MAX_FEATURES = 5000     # vocabulary size cap for both BoW and TF-IDF
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ------------------------------------------------------------------

df = pd.read_csv("cleaned_data.csv")
df = df.dropna(subset=["clean_text", "emotion"])

print("Loaded cleaned data:", df.shape)

# Encode emotion labels (e.g. joy -> 0, sadness -> 1, ...)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["emotion"])
joblib.dump(label_encoder, "label_encoder.pkl")
print("Classes:", list(label_encoder.classes_))

# Split BEFORE vectorizing so the vectorizer is fit only on training text
# (avoids data leakage from the test set into the vocabulary/IDF weights)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ------------------------------------------------------------------
# Bag-of-Words
# ------------------------------------------------------------------
bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)

joblib.dump(bow_vectorizer, "bow_vectorizer.pkl")
joblib.dump((X_train_bow, X_test_bow, y_train, y_test), "bow_features.pkl")
print("Bag-of-Words input shape:", X_train_bow.shape)

# ------------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), "tfidf_features.pkl")
print("TF-IDF input shape:", X_train_tfidf.shape)

print("\nFeature extraction complete. Saved:")
print(" - bow_features.pkl   (Bag-of-Words input layer)")
print(" - tfidf_features.pkl (TF-IDF input layer)")
print(" - bow_vectorizer.pkl / tfidf_vectorizer.pkl (needed later for new predictions)")
print(" - label_encoder.pkl")
