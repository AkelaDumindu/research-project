import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os

# --- 1. Robust Path Configuration ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from 'scripts')
project_root = os.path.dirname(script_dir)

# Build paths from the project root
DATA_PATH = os.path.join(project_root, "data", "combined_dataset.csv")
MODEL_PATH = os.path.join(project_root, "models", "logistic_regression_model.joblib")
RESULT_PATH = os.path.join(project_root, "results", "logistic_regression_report.txt")
# It's good practice to save the vectorizer and encoder used with the model
VECTORIZER_PATH = os.path.join(project_root, "models", "tfidf_vectorizer.joblib")
ENCODER_PATH = os.path.join(project_root, "models", "label_encoder.joblib")

# Create directories if they don't exist
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
os.makedirs(os.path.join(project_root, "results"), exist_ok=True)

# --- 2. Load and Preprocess Data ---
print("Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["text", "assignee"], inplace=True)

# --- 3. Filter Out Rare Classes (Fix for Stratify Error) ---
assignee_counts = df["assignee"].value_counts()
valid_assignees = assignee_counts[assignee_counts >= 2].index
df = df[df["assignee"].isin(valid_assignees)]
print(f"Data loaded. Found {df.shape[0]} valid records.")

# --- 4. Split Data with Stratification ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["assignee"],
    test_size=0.2,
    random_state=42,
    stratify=df["assignee"]
)

# --- 5. Feature Engineering and Label Encoding ---
print("Vectorizing text data and encoding labels...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# --- 6. Train the Model ---
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
model.fit(X_train_vec, y_train_enc)

# --- 7. Evaluate the Model ---
print("Evaluating the model...")
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred, average="weighted")

# --- 8. Save Results and Model ---
print("Saving results and model file...")
with open(RESULT_PATH, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nF1-score (weighted): {f1:.4f}\n\n")
    # Fix for classification_report error
    f.write(classification_report(
        y_test_enc,
        y_pred,
        target_names=encoder.classes_,
        labels=range(len(encoder.classes_)),
        zero_division=0
    ))

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(encoder, ENCODER_PATH)

print("---")
print(f"✅ Logistic Regression training complete!")
print(f"   Accuracy: {acc:.3f}")
print(f"   F1-score: {f1:.3f}")
print(f"   Model saved to: {MODEL_PATH}")
print(f"   Report saved to: {RESULT_PATH}")