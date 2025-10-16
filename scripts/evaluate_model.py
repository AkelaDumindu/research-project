# scripts/evaluate_model.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- helpers: MAP@K and MRR ---
def apk(actual, predicted, k=5):
    """
    actual: list of relevant labels (for us single-element lists, e.g. ['alice'])
    predicted: list of predicted labels (strings) ordered by score
    """
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    found = 0.0
    for i, p in enumerate(predicted, start=1):
        if p in actual and p not in predicted[:i-1]:
            found += 1.0
            score += found / i
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k=5):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])

def mrr_single(actual, predicted):
    # actual: single label string or list
    if isinstance(actual, (list, tuple)):
        target = actual[0] if actual else None
    else:
        target = actual
    for i, p in enumerate(predicted, start=1):
        if p == target:
            return 1.0 / i
    return 0.0

# --- paths (adjust if needed) ---
BASE = os.path.dirname(os.path.dirname(__file__))  # project root (scripts/..)
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "data")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_repo_model.joblib")
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
DATA_PATH = os.path.join(DATA_DIR, "combined_dataset.csv")

# --- load artifacts ---
print("Loading model / vectorizer / label encoder...")
model = joblib.load(MODEL_PATH)
vec = joblib.load(VECT_PATH)
le = joblib.load(LE_PATH)

# --- load dataset ---
print(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, dtype={"assignee": str, "text": str})  # ensure strings

# remove rows with null assignee or empty text
before_count = len(df)
df = df[df['assignee'].notna() & df['text'].notna()].copy()
after_nonnull = len(df)
print(f"Rows total: {before_count}, after dropping null assignee/text: {after_nonnull}")

# --- Detect unseen labels ---
known_labels = set(le.classes_.tolist())
df['assignee_str'] = df['assignee'].astype(str).str.strip()

mask_known = df['assignee_str'].isin(known_labels)
unknown_mask = ~mask_known
num_unknown = unknown_mask.sum()

if num_unknown > 0:
    print(f"⚠️ Found {num_unknown} rows with assignees unseen by the label-encoder. These will be dropped for evaluation.")
    # save the dropped rows for inspection
    dropped_df = df[unknown_mask].copy()
    dropped_path = os.path.join(LOG_DIR, "dropped_unseen_labels.csv")
    dropped_df.to_csv(dropped_path, index=False)
    print(f"  Saved dropped rows to {dropped_path}")
# filter dataset to known rows
df_known = df[mask_known].copy()
print(f"Evaluating on {len(df_known)} rows (dropped {num_unknown}).")

if len(df_known) == 0:
    raise SystemExit("No evaluation rows remained after filtering unseen labels. Either retrain label encoder or inspect dropped_unseen_labels.csv")

# --- Prepare X and true labels ---
X = vec.transform(df_known['text'].fillna(''))
# y_true as labels (strings) and encoded integers:
y_true_str = df_known['assignee_str'].tolist()
y_true_enc = le.transform(y_true_str)  # now safe because all are known

# --- Predict probabilities --- 
print("Predicting probabilities (this may take a while for large datasets)...")
probs = model.predict_proba(X)  # shape (n_samples, n_classes)

# --- Build top-K predicted label lists (strings) ---
K = 5
# top K indices per row (highest first)
topk_idx = np.argsort(probs, axis=1)[:, -K:][:, ::-1]  # shape (n_samples, K)

# inverse transform indices -> label strings
# sklearn's LabelEncoder.inverse_transform expects 1d array, but we can map indices to classes_ directly
classes_arr = np.array(le.classes_)
topk_labels = [[classes_arr[idx] for idx in row] for row in topk_idx]

# --- actual labels as list of single-element lists for MAP@K ---
actual_labels = [[lbl] for lbl in y_true_str]

# --- Metrics ---
map5 = mapk(actual_labels, topk_labels, k=K)
# MRR
mrr_vals = [mrr_single(a[0], pred) for a, pred in zip(actual_labels, topk_labels)]
mrr = float(np.mean(mrr_vals))

# Top-1 accuracy (straightforward)
top1_preds = [row[0] for row in topk_labels]
top1_acc = float(np.mean([1 if a==p else 0 for a,p in zip(y_true_str, top1_preds)]))

print("\n--- Evaluation results ---")
print(f"Evaluated rows: {len(df_known)}")
print(f"MAP@{K}: {map5:.4f}")
print(f"MRR: {mrr:.4f}")
print(f"Top-1 accuracy: {top1_acc:.4f}")

# Optionally: save per-row predictions for inspection
output_preds = df_known[['assignee_str']].copy()
output_preds['top5'] = ["|".join(row) for row in topk_labels]
output_preds['pred_top1'] = top1_preds
outpath = os.path.join(LOG_DIR, "predictions_with_top5.csv")
output_preds.to_csv(outpath, index=False)
print(f"Saved predictions (top5) to: {outpath}")
