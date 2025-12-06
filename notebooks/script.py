import argparse
import os
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # ---------- 1. Parse SageMaker Arguments ----------
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)

    # Directories and files
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", ""))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()

    # ---------- 2. Display environment info ----------
    print("📦 SKLearn Version:", sklearn.__version__)
    print("📦 Joblib Version:", joblib.__version__)
    print("📁 Training directory:", args.train)
    print("📁 Testing directory:", args.test)

    # ---------- 3. Read training and test data ----------
    print("[INFO] Reading data...\n")
    train_path = os.path.join(args.train, args.train_file)
    test_path = os.path.join(args.test, args.test_file)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("✅ Train shape:", train_df.shape)
    print("✅ Test shape:", test_df.shape)

    # ---------- 4. Encode labels and prepare features ----------
    le = LabelEncoder()
    train_df["flair"] = le.fit_transform(train_df["flair"].astype(str))
    test_df["flair"] = le.transform(test_df["flair"].astype(str))

    X_train = train_df.drop(columns=["flair"], errors="ignore")
    y_train = train_df["flair"]
    X_test = test_df.drop(columns=["flair"], errors="ignore")
    y_test = test_df["flair"]

    print("Number of features in training:", X_train.shape[1])
    print("Number of features in testing:", X_test.shape[1])

    # ---------- 5. Train Random Forest ----------
    print("🚀 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ---------- 6. Evaluate ----------
    print("📊 Generating classification report...\n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # ---------- 7. Save trained model and encoder ----------
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    encoder_path = os.path.join(args.model_dir, "label_encoder.joblib")
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"✅ Model saved at: {model_path}")
    print(f"✅ LabelEncoder saved at: {encoder_path}")



def input_fn(request_body, request_content_type):
    """Deserialize input data"""
    import io
    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return df.fillna(0)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction"""
    model = model_dict["model"]
    preds = model.predict(input_data)
    return preds  # numeric predictions

def output_fn(prediction, response_content_type):
    """Serialize prediction output"""
    if response_content_type == "text/csv":
        return ",".join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
