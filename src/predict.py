import os
import joblib
import pandas as pd
import scipy.sparse as sp

# --- Paths ---
MODEL_DIR = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\models"
DATA_PATH = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\data\processed\reddit_features_clean.csv"

# Choose model to use
MODEL_NAME = "LogisticRegression_model.joblib"   # change if needed

model_path = os.path.join(MODEL_DIR, MODEL_NAME)
encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# --- Load model, encoder, and vectorizer ---
print(f"Loading model from: {model_path}")
logistic_model = joblib.load(model_path)

print(f"Loading label encoder from: {encoder_path}")
label_encoder = joblib.load(encoder_path)

print(f"Loading TF-IDF vectorizer from: {vectorizer_path}")
vectorizer = joblib.load(vectorizer_path)

# --- Load dataset for test input ---
print("Loading dataset for test input...")
df = pd.read_csv(DATA_PATH)

# Use only text column for TF-IDF (ignore numeric for now)
if 'title' not in df.columns:
    raise ValueError("Dataset must contain a 'title' column for text input.")

titles = df['title'].fillna('').tolist()

# Transform using TF-IDF vectorizer
X_tfidf = vectorizer.transform(titles)

# ====== PADDING FIX ======
expected_features = logistic_model.coef_.shape[1]
if X_tfidf.shape[1] < expected_features:
    diff = expected_features - X_tfidf.shape[1]
    print(f"[INFO] Padding TF-IDF vector: adding {diff} empty columns to match expected {expected_features} features.")
    X_tfidf = sp.hstack([X_tfidf, sp.csr_matrix((X_tfidf.shape[0], diff))])
elif X_tfidf.shape[1] > expected_features:
    print(f"[INFO] Truncating TF-IDF vector: keeping only {expected_features} features.")
    X_tfidf = X_tfidf[:, :expected_features]

# --- Predict on few random samples ---
sample_idx = df.sample(5, random_state=42).index
X_sample = X_tfidf[sample_idx]
y_true = df.loc[sample_idx, 'flair']

print("\nMaking predictions on 5 random samples...")
pred_encoded = logistic_model.predict(X_sample)
y_pred = label_encoder.inverse_transform(pred_encoded)

# --- Display results ---
for i, (true_label, pred_label, title) in enumerate(zip(y_true, y_pred, df.loc[sample_idx, 'title'])):
    print(f"\nSample {i+1}")
    print(f"Title: {title}")
    print(f"True Flair: {true_label}")
    print(f"Predicted Flair: {pred_label}")

print("\n✅ Prediction completed (with padding fix applied).")
