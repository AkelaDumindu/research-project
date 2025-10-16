import joblib
import sys

vectorizer = joblib.load("../models/tfidf_vectorizer.joblib")
model = joblib.load("../models/xgb_repo_model.joblib")
le = joblib.load("../models/label_encoder.joblib")

issue_text = input("Enter issue title/description: ")

X = vectorizer.transform([issue_text])
probs = model.predict_proba(X)[0]
top_k = probs.argsort()[-5:][::-1]
recommendations = le.inverse_transform(top_k)

print("\nTop-5 recommended developers:")
for rank, dev in enumerate(recommendations, 1):
    print(f"{rank}. {dev}  (confidence: {probs[top_k[rank-1]]:.2f})")
