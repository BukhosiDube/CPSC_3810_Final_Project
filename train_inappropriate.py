import os
import joblib
import pandas as pd
import kagglehub

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
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "inappropriate_classifier.pkl")

path = kagglehub.dataset_download("muhammadatef/english-profanity-words-dataset")


def find_csv_file(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    raise FileNotFoundError("No CSV file found in downloaded Kaggle dataset.")


def load_data(folder_path):
    csv_path = find_csv_file(folder_path)
    print("Using CSV:", csv_path)

    data_frame = pd.read_csv(csv_path)

    data_frame = data_frame.dropna(subset=["text", "is_offensive"])
    data_frame["text"] = data_frame["text"].astype(str)
    data_frame["is_offensive"] = data_frame["is_offensive"].astype(int)

    return data_frame["text"].tolist(), data_frame["is_offensive"].tolist()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    texts, labels = load_data(path)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.1,
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