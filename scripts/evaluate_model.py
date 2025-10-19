# scripts/evaluate_models.py
import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ------------------- Helper metrics -------------------
def apk(actual, predicted, k=5):
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score, found = 0.0, 0.0
    for i, p in enumerate(predicted, start=1):
        if p in actual and p not in predicted[:i-1]:
            found += 1.0
            score += found / i
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k=5):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])

def mrr_single(actual, predicted):
    if isinstance(actual, (list, tuple)):
        target = actual[0] if actual else None
    else:
        target = actual
    for i, p in enumerate(predicted, start=1):
        if p == target:
            return 1.0 / i
    return 0.0

# ------------------- Paths -------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "combined_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Shared artifacts
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# Models to evaluate
MODELS = {
    "LogisticRegression": "logistic_regression_model.joblib",
    "RandomForest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib",
    "SVM": "svm_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
}

# ------------------- Load vectorizer & label encoder -------------------
print("📦 Loading shared vectorizer and label encoder...")
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ------------------- Load dataset -------------------
print(f"📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, dtype={"assignee": str, "text": str})
df = df[df["assignee"].notna() & df["text"].notna()].copy()

known_labels = set(label_encoder.classes_)
df = df[df["assignee"].isin(known_labels)].copy()

X = vectorizer.transform(df["text"].fillna(""))
y_true_str = df["assignee"].tolist()
y_true_enc = label_encoder.transform(y_true_str)

# ------------------- Evaluate all models -------------------
K = 5
results_summary = []

for name, model_file in MODELS.items():
    path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(path):
        print(f"⚠️ Skipping {name} (model not found).")
        continue

    print(f"\n🚀 Evaluating {name}...")
    model = joblib.load(path)

    # --- Handle models without predict_proba ---
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        # Convert decision_function outputs to probability-like ranking
        scores = model.decision_function(X)
        if scores.ndim == 1:  # binary
            probs = np.vstack([1 - scores, scores]).T
        else:
            probs = scores
    else:
        preds = model.predict(X)
        probs = np.zeros((len(preds), len(label_encoder.classes_)))
        for i, p in enumerate(preds):
            probs[i, p] = 1.0

    # --- Top-K predicted labels ---
    topk_idx = np.argsort(probs, axis=1)[:, -K:][:, ::-1]
    classes_arr = np.array(label_encoder.classes_)
    topk_labels = [[classes_arr[idx] for idx in row] for row in topk_idx]
    actual_labels = [[lbl] for lbl in y_true_str]

    # --- Metrics ---
    map5 = mapk(actual_labels, topk_labels, k=K)
    mrr_vals = [mrr_single(a[0], pred) for a, pred in zip(actual_labels, topk_labels)]
    mrr = float(np.mean(mrr_vals))
    top1_preds = [row[0] for row in topk_labels]
    top1_acc = accuracy_score(y_true_str, top1_preds)

    results_summary.append({
        "Model": name,
        f"MAP@{K}": round(map5, 4),
        "MRR": round(mrr, 4),
        "Top-1 Accuracy": round(top1_acc, 4),
    })

    print(f"✅ {name} Results:")
    print(f"   MAP@{K}: {map5:.4f} | MRR: {mrr:.4f} | Top-1 Accuracy: {top1_acc:.4f}")

# ------------------- Save summary -------------------
summary_df = pd.DataFrame(results_summary)
summary_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\n📊 Model comparison saved to {summary_csv}\n")
print(summary_df)
