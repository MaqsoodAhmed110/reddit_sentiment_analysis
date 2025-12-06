"""
Reddit Sentiment Visualization Script
-------------------------------------
Performs:
- Feature correlation heatmap
- PCA (2D)
- t-SNE (2D)
Saves plots to ../plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# ---------- 1. Load Data ----------
data_path = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\data\processed\reddit_features.csv"
df = pd.read_csv(data_path)
print("✅ Data Loaded:", df.shape)

# ---------- 2. Prepare Output Directory ----------
plots_dir = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\plots"
os.makedirs(plots_dir, exist_ok=True)

# ---------- 3. Prepare Data for Visualization ----------
# Extract numeric feature columns (TF-IDF features)
feature_cols = df.columns[2:]  # first two are 'title', 'flair'
X = df[feature_cols].values
labels = df['flair']

# ---------- 4. Correlation Heatmap ----------
plt.figure(figsize=(12, 8))
corr = pd.DataFrame(X, columns=feature_cols).corr()
sns.heatmap(corr, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title("Feature Correlation Heatmap (TF-IDF)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "heatmap.png"))
plt.close()
print("✅ Saved: heatmap.png")

# ---------- 5. PCA Visualization ----------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=60, alpha=0.8)
plt.title("PCA - Reddit TF-IDF Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Flair", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "pca_plot.png"))
plt.close()
print("✅ Saved: pca_plot.png")

# ---------- 6. t-SNE Visualization ----------
tsne = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate='auto')
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="tab10", s=60, alpha=0.8)
plt.title("t-SNE - Reddit TF-IDF Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Flair", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "tsne_plot.png"))
plt.close()
print("✅ Saved: tsne_plot.png")

print("\n🎉 Visualization complete! All plots saved to:", plots_dir)
