# Self-Moderating Machine Learning Model for EdTech
**CPSC 3810: Introduction to Machine Learning — Final Project**
**Author: Bukhosi Dube**

---

## Overview

This project implements an automated moderation system for educational discussion
platforms (e.g. Ed Discussion, Piazza). Given a CSV of student posts, the system
classifies each post as appropriate/relevant and outputs only the posts that should
be kept, filtering out inappropriate or irrelevant content.

The pipeline uses sentence embeddings (`all-MiniLM-L6-v2`) combined with two
logistic regression classifiers — one for inappropriate language detection and one
for course relevance detection.

---

## Project Structure

```
CPSC_3810_Final_Project/
├── moderation_script.py          # Main script — run this to moderate posts
├── train_inappropriate.py        # (Optional) retrain the inappropriate classifier
├── train_irrelevant.py            # (Optional) retrain the relevance classifier
├── requirements.txt           
│
├── models/
│   ├── inappropriate_classifier.pkl   # Pre-trained model (no retraining needed)
│   └── relevance_classifier.pkl       # Pre-trained model (no retraining needed)
│
├── data/
│   ├── edtech_test_dataset.csv              # Small test dataset
│   ├── edtech_test_dataset_large.csv        # Large test dataset (default input)
│   ├── filtered_relevant_questions.csv      # Example output
│   └── relevance_dataset_cpsc3810.csv       # Training data for relevance model
│
└── comparison_results/
    ├── compare_models.py                    # Utility: compare two model outputs
    ├── filtered_relevant_questions.csv      # LogReg output used for comparison
    ├── filtered_relevant_questions_generated.csv  # GPT output used for comparison
    └── results/                             # Generated charts and stats
```

---

## Requirements

**Python version:** 3.9 or higher

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas
scikit-learn
sentence-transformers
joblib
```

> Note: `sentence-transformers` will automatically download the
> `all-MiniLM-L6-v2` embedding model on first run (~80MB). An internet
> connection is required the first time only.

---

## How to Run

### Step 1 — Moderate a dataset (main usage)

From the project root directory, run:

```bash
python moderation_script.py
```

This will:
- Load posts from `data/edtech_test_dataset_large.csv`
- Run each post through the inappropriate and relevance classifiers
- Save posts marked as KEEP to `comparison_results/filtered_relevant_questions.csv`

**Expected output:**
```
Saved kept relevant questions to comparison_results/filtered_relevant_questions.csv
```

### Step 2 — (Optional) Compare against an LLM output

If you have a second CSV of classifications from an LLM (e.g. GPT), you can
compare the two outputs and generate statistics:

```bash
cd comparison_results
python compare_models.py --file1 filtered_relevant_questions.csv --file2 filtered_relevant_questions_generated.csv --name1 "LogReg" --name2 "GPT" --output ./results
```

This generates agreement stats, confusion matrices, and disagreement CSVs
inside `comparison_results/results/`.

---

## Pre-trained Models

The `models/` directory already contains trained `.pkl` files — **no retraining
is required to run the system.** 

### (Optional) Retraining from scratch

If you wish to retrain the models yourself:

**Inappropriate language classifier** (downloads dataset from Kaggle automatically):
```bash
python train_inappropriate.py
```

**Relevance classifier** (uses `data/relevance_dataset_cpsc3810.csv`):
```bash
python train_irrelevant.py
```

> Warning: Retraining will overwrite the existing `.pkl` files in `models/`.

---

## Input Format

The input CSV must contain a column named `text`. Example:

| text |
|------|
| How does gradient descent work? |
| Can you explain regularization? |

---

## Output Format

The output CSV contains one row per post that passed moderation:

| text | is_inappropriate | is_relevant | decision |
|------|-----------------|-------------|----------|
| How does gradient descent work? | 0 | 1 | KEEP |

---

## Notes

- The system is scoped to **CPSC 3810: Introduction to Machine Learning** posts.
  Relevance is defined relative to that course context.
- The relevance classifier uses a **probability threshold of 0.6** — posts must
  exceed 60% confidence to be marked as KEEP.
