# scripts/train_svm.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib, os

DATA_PATH = "data/combined_dataset.csv"
MODEL_PATH = "models/svm_model.joblib"
RESULT_PATH = "results/svm_report.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.dropna(subset=["text", "assignee"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["assignee"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

model = LinearSVC()
model.fit(X_train_vec, y_train_enc)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred, average="weighted")

with open(RESULT_PATH, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nF1-score: {f1:.4f}\n")
    f.write(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

joblib.dump(model, MODEL_PATH)
print(f"✅ SVM trained. Accuracy={acc:.3f}, F1={f1:.3f}")
