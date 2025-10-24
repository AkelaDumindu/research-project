import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os

DATA_FILE = "../data/combined_dataset.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df[df['assignee'].notnull()]

# Combine title + body
df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')

# Limit to frequent developers
counts = df['assignee'].value_counts()
df = df[df['assignee'].isin(counts[counts >= 5].index)]

# Encode text and labels
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
le = LabelEncoder()
y = le.fit_transform(df['assignee'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Model
model = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', n_estimators=300)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.joblib")) 
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print("✅ Model saved successfully.")
