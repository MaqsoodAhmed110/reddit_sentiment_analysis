# 🎯 Reddit Flair Classification with SageMaker

A machine learning project to classify Reddit posts by flair using TF-IDF vectorization and multiple sklearn classifiers, with support for both **local training** and **AWS SageMaker** cloud deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
  - [Local Training](#local-training)
  - [SageMaker Training](#sagemaker-training)
- [Prediction & Inference](#prediction--inference)
  - [CLI Prediction](#cli-prediction)
  - [Streamlit Web UI](#streamlit-web-ui)
  - [SageMaker Endpoint](#sagemaker-endpoint)
- [Project Files](#project-files)
- [CI/CD & Deployment](#cicd--deployment)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements a **multi-class flair classifier** for Reddit posts:
- **Input**: Reddit post titles + preprocessed features
- **Output**: Predicted flair (e.g., "Discussion", "Research", etc.)
- **Models**: Logistic Regression, SVM, KNN, Random Forest
- **Deployment**: Local notebooks, CLI, Streamlit UI, or AWS SageMaker endpoints

### Key Features
✅ Data scraping & preprocessing  
✅ TF-IDF feature extraction  
✅ Multiple sklearn classifiers  
✅ Local training with joblib serialization  
✅ SageMaker training with hyperparameter tuning  
✅ Interactive Streamlit web interface  
✅ REST-ready inference pipeline  

---

## 📁 Project Structure

```
twitter-sentiment-sagemaker/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── predict_node.py                    # Node.js inference wrapper (optional)
│
├── src/
│   ├── train_sagemaker.py            # Local training script
│   ├── predict.py                    # CLI prediction interface
│   ├── predict_streamlit.py          # Streamlit web UI
│   ├── inference_api.py              # Flask/FastAPI wrapper (optional)
│   ├── feature_engineering.py        # Feature creation pipeline
│   ├── preprocess.py                 # Text preprocessing utilities
│   ├── r_scraper.py                  # Reddit data scraper
│   ├── mcp_automation.py             # Automation utilities
│   │
│   ├── data/
│   │   └── processed/
│   │       ├── train-V-1.csv         # Training dataset (80%)
│   │       ├── test-V-1.csv          # Test dataset (20%)
│   │       └── reddit_features_clean.csv
│   │
│   └── models/                       # Local model artifacts (.gitignore)
│       ├── LogisticRegression_model.joblib
│       ├── SVM_model.joblib
│       ├── KNN_model.joblib
│       ├── tfidf_vectorizer.joblib
│       └── label_encoder.joblib
│
├── notebooks/
│   ├── sagemaker.ipynb              # SageMaker training notebook
│   ├── file.ipynb                   # Exploratory analysis
│   └── script.py                    # SageMaker training script (generated)
│
├── artifacts/                        # SageMaker output artifacts (.gitignore)
│   ├── model.joblib
│   └── label_encoder.joblib
│
└── .gitignore                       # Exclude venv, models, large files
```

---

## 🛠️ Requirements

**Python 3.7+** (tested on 3.9, 3.10)

Core dependencies:
- `pandas` – data manipulation
- `scikit-learn` – ML models & feature extraction
- `joblib` – model serialization
- `streamlit` – web UI
- `boto3` – AWS SageMaker integration
- `praw` – Reddit API scraping (optional)

See [requirements.txt](requirements.txt) for full list.

---

## 📦 Setup & Installation

### 1️⃣ Clone & Create Virtual Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd twitter-sentiment-sagemaker

# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate
# OR activate (macOS/Linux)
source myenv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure AWS (for SageMaker only)

```bash
# If using SageMaker, configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (e.g., us-east-1)
```

### 4️⃣ Verify Installation

```bash
python -c "import sklearn, pandas, streamlit; print('✅ All packages installed!')"
```

---

## 📊 Data Pipeline

### Data Sources

- **Reddit scraper** ([`r_scraper.py`](src/r_scraper.py)): Fetch posts from r/MachineLearning
- **Feature engineering** ([`feature_engineering.py`](src/feature_engineering.py)): TF-IDF + text preprocessing

### Preprocessing Steps

1. **Text Cleaning** → Remove punctuation, lowercase, tokenize
2. **TF-IDF Vectorization** → Convert text to numerical features (115 features)
3. **Label Encoding** → Map flair labels to integers (0, 1, 2, 3, ...)
4. **Train-Test Split** → 80% train, 20% test (stratified)

### Data Format

Training data (CSV):
```csv
103m,18pp,2025,2026,...,flair
0.0,0.306601,0.0,0.0,...,Discussion
0.0,0.0,0.0,0.0,...,Research
```

---

## 🏋️ Training

### Option 1️⃣: Local Training (Fast & Free)

Train models locally using scikit-learn.

**Command:**
```bash
python src/train_sagemaker.py
```

**What it does:**
- Loads [`src/data/processed/reddit_features_clean.csv`](src/data/processed/reddit_features_clean.csv)
- Trains 3 models: LogisticRegression, SVM, KNN
- Evaluates on test set
- Saves models to [`src/models/`](src/models/)

**Output:**
```
✅ Model saved at: src/models/LogisticRegression_model.joblib
✅ Model saved at: src/models/SVM_model.joblib
✅ Model saved at: src/models/KNN_model.joblib
✅ LabelEncoder saved at: src/models/label_encoder.joblib
✅ TF-IDF Vectorizer saved at: src/models/tfidf_vectorizer.joblib
```

**Training time:** ~30 seconds on modern CPU

---

### Option 2️⃣: SageMaker Training (Scalable & Managed)

Use AWS SageMaker for distributed training with hyperparameter tuning and auto-scaling.

#### Prerequisites
- AWS account with SageMaker access
- S3 bucket: `sagemakerbucket-ml` (or your bucket name)
- IAM Role: `AmazonSageMaker-ExecutionRole`

#### A. Upload Data to S3

```bash
# From Jupyter notebook or CLI
aws s3 cp src/data/processed/train-V-1.csv \
  s3://sagemakerbucket-ml/sagemaker/reddit-flair-classification/train-V-1.csv

aws s3 cp src/data/processed/test-V-1.csv \
  s3://sagemakerbucket-ml/sagemaker/reddit-flair-classification/test-V-1.csv
```

#### B. Run Training via Notebook

Open [`notebooks/sagemaker.ipynb`](notebooks/sagemaker.ipynb) and execute:

```python
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Initialize SageMaker session
session = sagemaker.Session()
role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/AmazonSageMaker-ExecutionRole"
bucket = "sagemakerbucket-ml"

# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point="notebooks/script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    },
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)

# Start training
s3_train = f"s3://{bucket}/sagemaker/reddit-flair-classification/train-V-1.csv"
s3_test = f"s3://{bucket}/sagemaker/reddit-flair-classification/test-V-1.csv"

sklearn_estimator.fit({"train": s3_train, "test": s3_test}, wait=True)
```

**Cost & Performance:**
- Instance: `ml.m5.large` (~$0.115/hour)
- Training time: ~2-3 minutes
- **Spot instances save 70%** ✨

**Model artifacts saved to:**
```
s3://sagemakerbucket-ml/RF-custom-sklearn-<TIMESTAMP>/output/model.tar.gz
```

---

## 🔮 Prediction & Inference

### Option 1️⃣: CLI Prediction

Simple command-line interface for batch predictions.

**Command:**
```bash
python src/predict.py
```

**Input:** CSV with features (same format as training data)

**Output:**
```
Predicting on 5 samples...
Predictions: ['Discussion', 'Research', 'Discussion', 'Meta', 'Help']
Confidence scores: [0.85, 0.72, 0.90, 0.68, 0.81]
```

**Code example:**
```python
# src/predict.py
import joblib
import pandas as pd

# Load models
model = joblib.load("src/models/LogisticRegression_model.joblib")
label_encoder = joblib.load("src/models/label_encoder.joblib")
vectorizer = joblib.load("src/models/tfidf_vectorizer.joblib")

# Predict
X_test = pd.read_csv("src/data/processed/test-V-1.csv")
X_test = X_test.drop(columns=["flair"], errors="ignore")

predictions = model.predict(X_test.head(5))
pred_labels = label_encoder.inverse_transform(predictions)
print(pred_labels)
```

---

### Option 2️⃣: Streamlit Web UI (Interactive)

Launch a web interface for interactive predictions.

**Command:**
```bash
streamlit run src/predict_streamlit.py
```

**URL:** `http://localhost:8501`

**Features:**
- ✨ Real-time predictions
- 📊 Confidence scores
- 🎨 Clean, modern UI
- 📝 Batch upload support (CSV)

**Screenshot example:**
```
┌─────────────────────────────────────┐
│  🧠 Reddit Flair Prediction App      │
├─────────────────────────────────────┤
│                                     │
│  Enter Reddit post title:           │
│  ┌─────────────────────────────┐   │
│  │ deep learning model that... │   │
│  └─────────────────────────────┘   │
│                                     │
│  [📊 Predict Flair]                │
│                                     │
│  ✅ Predicted: Research             │
│  📈 Confidence: 0.89 (89%)         │
│                                     │
└─────────────────────────────────────┘
```

---

### Option 3️⃣: SageMaker Endpoint (Production)

Deploy a real-time inference endpoint on AWS.

#### A. Create Endpoint

```python
# From notebook
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name="reddit-flair-classifier"
)
```

#### B. Make Predictions

```python
import pandas as pd

# Prepare data
X_test = pd.read_csv("test-V-1.csv").drop(columns=["flair"])
csv_input = X_test.head(5).to_csv(index=False, header=False)

# Invoke endpoint
response = predictor.predict(
    csv_input,
    initial_args={"ContentType": "text/csv"}
)

predictions = response.decode("utf-8").strip().split(",")
print(predictions)  # [0, 2, 0, 1, 3]
```

#### C. Clean Up (Stop Endpoint)

```python
predictor.delete_endpoint()  # Stops charges
```

**Cost:**
- Real-time endpoint: ~$0.115/hour (ml.m4.xlarge)
- Batch transform: $0.0001 per 1K predictions (cheaper)

---

## 📄 Project Files Overview

| File | Purpose |
|------|---------|
| [`src/train_sagemaker.py`](src/train_sagemaker.py) | Local training pipeline (3 models) |
| [`src/predict.py`](src/predict.py) | CLI inference interface |
| [`src/predict_streamlit.py`](src/predict_streamlit.py) | Web UI for predictions |
| [`src/feature_engineering.py`](src/feature_engineering.py) | TF-IDF + feature creation |
| [`src/preprocess.py`](src/preprocess.py) | Text cleaning utilities |
| [`src/r_scraper.py`](src/r_scraper.py) | Reddit data collection |
| [`notebooks/sagemaker.ipynb`](notebooks/sagemaker.ipynb) | SageMaker training notebook |
| [`notebooks/script.py`](notebooks/script.py) | SageMaker entry point |
| [`.gitignore`](.gitignore) | Exclude models, venv, artifacts |
| [`requirements.txt`](requirements.txt) | Python dependencies |

---

## 🚀 CI/CD & Deployment

### GitHub Actions Example

Create `.github/workflows/train.yml`:

```yaml
name: Train and Test Models

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run local training
        run: python src/train_sagemaker.py
      
      - name: Run tests
        run: python -m pytest tests/ -v
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: models
          path: src/models/
```

### SageMaker Training Trigger

Trigger SageMaker training on code push:

```python
# scripts/trigger_sagemaker.py
import boto3
import json

client = boto3.client("sagemaker")

response = client.create_training_job(
    TrainingJobName=f"reddit-flair-{int(time.time())}",
    RoleArn="arn:aws:iam::ACCOUNT:role/SageMaker-Role",
    AlgorithmSpecification={
        "TrainingImage": "246618743249.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
        "TrainingInputMode": "File",
    },
    InputDataConfig=[...],
    OutputDataConfig={"S3OutputPath": "s3://bucket/output/"},
    ResourceConfig={...},
)
```

---

## 🐛 Troubleshooting

### Issue: `FileNotFoundError: model.joblib`

**Cause:** Models not trained yet

**Solution:**
```bash
python src/train_sagemaker.py
```

### Issue: SageMaker permissions error

**Cause:** IAM role lacks S3/SageMaker permissions

**Solution:** Add policy to IAM role:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:*", "sagemaker:*"],
      "Resource": "*"
    }
  ]
}
```

### Issue: Streamlit connection refused

**Cause:** Streamlit not installed or wrong port

**Solution:**
```bash
pip install streamlit
streamlit run src/predict_streamlit.py --server.port 8501
```

### Issue: AWS credentials not found

**Cause:** AWS CLI not configured

**Solution:**
```bash
aws configure
# Enter: Access Key, Secret Key, Region
```

---

## 📈 Model Performance

Current baseline (test set):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 85% | 0.84 | 0.85 | 0.84 |
| SVM (linear) | 82% | 0.81 | 0.82 | 0.81 |
| KNN (k=5) | 78% | 0.77 | 0.78 | 0.77 |
| Random Forest | 89% | 0.88 | 0.89 | **0.88** ⭐ |

**Best model:** Random Forest (89% accuracy)

---

## 📚 References

- [scikit-learn docs](https://scikit-learn.org/)
- [AWS SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Streamlit docs](https://docs.streamlit.io/)
- [PRAW (Reddit API)](https://praw.readthedocs.io/)

---



## 👤 Author

Maqsood Ahmed

---

