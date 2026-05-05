import os
import re
import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INAPPROPRIATE_MODEL_PATH = "models/inappropriate_classifier.pkl"
RELEVANCE_MODEL_PATH = "models/relevance_classifier.pkl"

INPUT_PATH = "data/edtech_test_dataset_large.csv"
OUTPUT_PATH = "data/filtered_relevant_questions.csv"


def preprocess(text):
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def moderate_text(text, embedder, inappropriate_model, relevance_model):
    cleaned_text = preprocess(text)

    embedding = embedder.encode([cleaned_text])

    is_inappropriate = inappropriate_model.predict(embedding)[0]

    if is_inappropriate == 1:
        return {
            "text": cleaned_text,
            "is_inappropriate": 1,
            "is_relevant": None,
            "decision": "REMOVED",
        }
    
    is_relevant = relevance_model.predict(embedding)[0]
    prob = relevance_model.predict_proba(embedding)[0][1]
    #print(cleaned_text, prob)

    if prob > 0.6:
        decision = "KEEP"
    else:
        decision = "REMOVED"

    return {
        "text": cleaned_text,
        "is_inappropriate": 0,
        "is_relevant": int(is_relevant),
        "decision": decision,
    }


def main():
    data_frame = pd.read_csv(INPUT_PATH)
    if "text" not in data_frame.columns:
        raise ValueError("Input CSV must have text")
    
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    inappropriate_model = joblib.load(INAPPROPRIATE_MODEL_PATH)
    relevance_model = joblib.load(RELEVANCE_MODEL_PATH)

    results = []
    for text in data_frame["text"]:
        result = moderate_text(text, embedder, inappropriate_model, relevance_model)
        results.append(result)

    results_data_frame = pd.DataFrame(results)
    kept_questions = results_data_frame[results_data_frame["decision"] == "KEEP"]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    kept_questions.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Saved kept relevant questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
