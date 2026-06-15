# Text Emotion Detection (NLTK + BoW/TF-IDF + Logistic Regression)

A simple, modular NLP pipeline for emotion detection from text, split into
separate steps so feature extraction ("input layer") is generated once and
reused for training/evaluation.

## ⚠️ About the dataset

You mentioned `ananthu017/emotion-detection-fer`, but that's the **FER-2013
facial expression dataset — it contains face images, not text**, so it
can't be used with NLTK / Bag-of-Words / TF-IDF / Logistic Regression.

This pipeline instead downloads **`praveengovi/emotions-dataset-for-nlp`**
(text sentences labeled with emotions: joy, sadness, anger, fear, love,
surprise) via `kagglehub` — same download mechanism, just a dataset that
actually fits this task.

If you have a different text dataset in mind, open `1_data_preprocessing.py`
and change:
```python
KAGGLE_DATASET = "praveengovi/emotions-dataset-for-nlp"
TEXT_COLUMN = "text"
LABEL_COLUMN = "emotion"
```
to match your dataset's Kaggle slug and column names.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. `kagglehub` needs your Kaggle API credentials. Either:
   - Place `kaggle.json` (from kaggle.com -> Account -> Create New API Token)
     in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows), **or**
   - Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Pipeline (run in order)

| Script | What it does | Output |
|---|---|---|
| `1_data_preprocessing.py` | Downloads dataset, applies lowercasing, special-char removal, tokenization, stopword removal, lemmatization | `cleaned_data.csv` |
| `2_feature_extraction.py` | Builds the **Bag-of-Words** and **TF-IDF** "input layers", does the train/test split | `bow_features.pkl`, `tfidf_features.pkl`, vectorizers, label encoder |
| `3_train_evaluate.py` | Trains Logistic Regression on each feature set, reports **F1 (macro/weighted)**, classification report, and saves **confusion matrix** plots | `logreg_model_BoW.pkl`, `logreg_model_TFIDF.pkl`, `confusion_matrix_*.png` |
| `4_predict.py` (optional) | Predicts emotion for new custom sentences using the saved TF-IDF model | console output |

Run them one at a time:
```bash
python 1_data_preprocessing.py
python 2_feature_extraction.py
python 3_train_evaluate.py
python 4_predict.py
```

## Notes / tuning

- `NORMALIZATION` in `1_data_preprocessing.py` can be switched between
  `"lemmatize"` (default, generally better) and `"stem"`.
- `MAX_FEATURES` in `2_feature_extraction.py` controls vocabulary size for
  both BoW and TF-IDF — increase it (e.g. 10000+) for slightly better
  accuracy at the cost of speed/memory.
- The TF-IDF model usually outperforms BoW for text classification, but
  the script trains and evaluates both so you can compare F1 scores
  directly.
- If you'd like a neural network (Keras/PyTorch) version instead of
  Logistic Regression, the same `bow_features.pkl` / `tfidf_features.pkl`
  outputs can be fed straight into a `Dense` input layer.
