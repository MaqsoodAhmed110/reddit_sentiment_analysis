import os
import joblib
import streamlit as st
import scipy.sparse as sp

# ===========================
# Setup and Model Loading
# ===========================
MODEL_DIR = r"C:\Users\user\Desktop\twitter-sentiment-sagemaker\src\models"

try:
    logistic_model = joblib.load(os.path.join(MODEL_DIR, "LogisticRegression_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

st.title("🧠 Reddit Flair Prediction App")
st.write("This app predicts the **flair** of a Reddit post title using a trained Logistic Regression model.")

# ===========================
# User Input Section
# ===========================
user_input = st.text_area("Enter the Reddit post title:", height=150)

# ===========================
# Prediction Logic
# ===========================
if st.button("Predict Flair"):
    if user_input.strip():
        try:
            # Transform input using the same TF-IDF vectorizer
            X_tfidf = vectorizer.transform([user_input])

            # Handle TF-IDF feature mismatch (if occurs)
            expected_features = logistic_model.coef_.shape[1]
            if X_tfidf.shape[1] < expected_features:
                diff = expected_features - X_tfidf.shape[1]
                X_tfidf = sp.hstack([X_tfidf, sp.csr_matrix((1, diff))])
            elif X_tfidf.shape[1] > expected_features:
                X_tfidf = X_tfidf[:, :expected_features]

            # Predict flair
            pred_encoded = logistic_model.predict(X_tfidf)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"**Predicted Flair:** 🎯 {pred_label}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please enter a Reddit post title before predicting!")

# ===========================
# Footer
# ===========================
st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF (max_features=200) | Built with ❤️ using Streamlit")
