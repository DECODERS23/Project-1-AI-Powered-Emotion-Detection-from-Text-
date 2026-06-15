"""
STEP 4 (optional) - Predict Emotion on New Text
=================================================
Loads the saved TF-IDF vectorizer + Logistic Regression model and predicts
the emotion of new, custom sentences. Reuses the exact same preprocessing
from Step 1.
"""

import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# Load the TF-IDF pipeline (swap to "BoW" / bow_vectorizer.pkl if you prefer that model)
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logreg_model_TFIDF.pkl")
label_encoder = joblib.load("label_encoder.pkl")


def predict_emotion(text: str) -> str:
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return label_encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    samples = [
        "I can't believe I won the lottery, this is amazing!",
        "I'm so scared of what might happen tomorrow.",
        "This traffic is making me so angry right now.",
        "I miss my old friends so much, feeling really down.",
    ]
    for s in samples:
        print(f"{s!r:60s} -> {predict_emotion(s)}")
