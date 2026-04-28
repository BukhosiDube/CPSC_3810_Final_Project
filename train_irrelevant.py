# train_relevance.py

import os
import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "relevance_dataset_cpsc3810.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "relevance_classifier.pkl")


def load_data(path):
    data_frame = pd.read_csv(path)

    data_frame = data_frame.dropna(subset=["text", "is_relevant"])
    data_frame["text"] = data_frame["text"].astype(str)
    data_frame["is_relevant"] = data_frame["is_relevant"].astype(int)

    print("Dataset shape:", data_frame.shape)
    print("Class counts:")
    print(data_frame["is_relevant"].value_counts())

    return data_frame["text"].tolist(), data_frame["is_relevant"].tolist()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    texts, labels = load_data(DATA_PATH)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    embedder_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    X_train = embedder_model.encode(X_train_text, show_progress_bar=True)
    X_test = embedder_model.encode(X_test_text, show_progress_bar=True)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(classifier, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()