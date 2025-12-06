import sys
import json
import os
import joblib
import pandas as pd

model_dir = sys.argv[1]
data_path = sys.argv[2]
model_name = sys.argv[3]
samples_json = sys.argv[4]

# Load model and encoder
model = joblib.load(os.path.join(model_dir, model_name))
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

# Load CSV and remove non-numeric columns except 'flair'
df = pd.read_csv(data_path)
non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
non_numeric = [c for c in non_numeric if c != 'flair']
if non_numeric:
    df = df.drop(columns=non_numeric)

# Prepare input
samples = pd.DataFrame(json.loads(samples_json))

# Predict
y_pred_encoded = model.predict(samples)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Return predictions as JSON
print(json.dumps(y_pred.tolist()))
