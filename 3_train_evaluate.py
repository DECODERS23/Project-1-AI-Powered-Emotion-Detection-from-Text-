"""
STEP 3 - Model Training & Evaluation
======================================
Loads the BoW and TF-IDF feature sets produced by 2_feature_extraction.py,
trains a separate Logistic Regression model on each, and evaluates with:

  - F1 score (macro and weighted)
  - Classification report (precision/recall/F1 per emotion)
  - Confusion matrix (saved as PNG)

Outputs:
  logreg_model_BoW.pkl, logreg_model_TFIDF.pkl
  confusion_matrix_BoW.png, confusion_matrix_TFIDF.png
"""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

label_encoder = joblib.load("label_encoder.pkl")
class_names = label_encoder.classes_


def train_and_evaluate(feature_file: str, name: str):
    X_train, X_test, y_train, y_test = joblib.load(feature_file)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n========== {name} ==========")
    print(f"F1 score (macro):    {f1_macro:.4f}")
    print(f"F1 score (weighted): {f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix plot
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


if __name__ == "__main__":
    results = {}
    results["BoW"] = train_and_evaluate("bow_features.pkl", "BoW")
    results["TFIDF"] = train_and_evaluate("tfidf_features.pkl", "TFIDF")

    print("\n========== Summary ==========")
    for name, (_, f1_macro, f1_weighted) in results.items():
        print(f"{name:6s} -> F1 macro: {f1_macro:.4f} | F1 weighted: {f1_weighted:.4f}")
