# scripts/recommend_issue_dev.py
"""
recommend_issue_dev.py
------------------------------------------------
Predicts top-5 developer recommendations for a given GitHub issue text
using the trained ML model, vectorizer, and label encoder.
"""

import joblib
import os
import numpy as np

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_repo_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# ------------------ Load Artifacts ------------------
try:
    print("📦 Loading model and artifacts...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Artifacts loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model/vectorizer/encoder: {e}")
    exit(1)

# ------------------ User Input ------------------
print("🔍 Enter the issue title/description below:")
issue_text = input("> ").strip()

if not issue_text:
    print("⚠️ No issue text provided. Exiting.")
    exit(0)

# ------------------ Transform Input ------------------
X = vectorizer.transform([issue_text])

# ------------------ Predict Top-5 Developers ------------------
try:
    probs = model.predict_proba(X)[0]
except AttributeError:
    print("⚠️ This model does not support probability predictions. Using decision function instead.")
    if hasattr(model, "decision_function"):
        probs = model.decision_function(X)[0]
        probs = np.exp(probs) / np.sum(np.exp(probs))  # softmax approximation
    else:
        print("❌ Model does not support predict_proba or decision_function.")
        exit(1)

# Get top-5 developer indices
top_k = probs.argsort()[-5:][::-1]
recommendations = label_encoder.inverse_transform(top_k)

# ------------------ Display Results ------------------
print("\n🎯 Top-5 Recommended Developers:")
for rank, dev in enumerate(recommendations, start=1):
    print(f"{rank}. {dev}  (confidence: {probs[top_k[rank-1]]:.3f})")

print("\n✅ Recommendation completed successfully!")
