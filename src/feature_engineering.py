import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

# ---------- 1. Load Data ----------
data_path = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\data\raw\reddit_data.csv"
df = pd.read_csv(data_path)
print("✅ Data Loaded:", df.shape)

# ---------- 2. Clean Text ----------
df['title'] = df['title'].fillna("").str.lower()
df['flair'] = df.get('flair', pd.Series(["None"] * len(df))).fillna("None")

# ---------- 3. Encode Flair as Label (Sentiment Proxy) ----------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['flair'])

# ---------- 4. Extract Text Features (TF-IDF) ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_tfidf = vectorizer.fit_transform(df['title']).toarray()
X = pd.DataFrame(X_tfidf, columns=vectorizer.get_feature_names_out())

# ---------- 5. Combine Features + Target ----------
df_final = pd.concat([df[['title', 'flair']], X], axis=1)

# ---------- 6. Save Output ----------
output_dir = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\data\processed"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "reddit_features.csv")
df_final.to_csv(output_path, index=False)

print(f"✅ Feature Engineering complete! File saved to {output_path}")
