import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# Paths
DATA_PATH = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\data\processed\reddit_features_clean.csv"
OUTPUT_DIR = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Ensure 'flair' column exists
if 'flair' not in df.columns:
    raise ValueError("The dataset must contain a 'flair' column for labels.")

# Apply TF-IDF vectorizer on 'title'
print("Applying TF-IDF vectorizer on 'title' column...")
tfidf_vectorizer = TfidfVectorizer(max_features=200)
X_tfidf = tfidf_vectorizer.fit_transform(df['title'].fillna(''))
tfidf_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
joblib.dump(tfidf_vectorizer, tfidf_path)
print(f"TF-IDF vectorizer saved at: {tfidf_path}")

# Combine TF-IDF features with remaining numeric features
numeric_features = df.drop(columns=['flair', 'title'])
X = pd.concat([pd.DataFrame(X_tfidf.toarray(), index=df.index), numeric_features.reset_index(drop=True)], axis=1)

# Convert all column names to strings to avoid sklearn feature name errors
X.columns = X.columns.astype(str)

y = df['flair']
print(f"Detected {X.shape[1]} total feature columns (TF-IDF + numeric).")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
label_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
joblib.dump(label_encoder, label_path)
print(f"Label encoder saved at: {label_path}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    print(f"Evaluating {name}...")
    y_pred = model.predict(X_test)

    labels = unique_labels(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[label_encoder.classes_[i] for i in labels],
        output_dict=True,
        zero_division=0
    )

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"{name} model saved at: {model_path}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, f"{name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"{name} classification report saved at: {report_path}")

print("\n✅ Training completed successfully for all models.")
print(f"All files saved under: {OUTPUT_DIR}")
