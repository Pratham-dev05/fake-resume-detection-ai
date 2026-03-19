import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
from PyPDF2 import PdfReader

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text
from src.features import extract_features, extract_features_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models/vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/resumes.csv")

# 🔥 AUTO TRAIN FUNCTION
def train_model():
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].astype(str)
    labels = df["label"]

    cleaned = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
    X_text = vectorizer.fit_transform(cleaned).toarray()

    X_features = [extract_features(t) for t in cleaned]

    X = np.hstack((X_text, X_features))

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, labels)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VEC_PATH, "wb"))

    return model, vectorizer

# 🔥 LOAD OR TRAIN
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VEC_PATH, "rb"))

    # check fitted
    if not hasattr(vectorizer, "idf_"):
        raise Exception("Not fitted")

except:
    st.warning("⚠️ Training model for first time...")
    model, vectorizer = train_model()
    st.success("✅ Model trained successfully!")

# UI
st.title("🤖 Fake Resume Detection System")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() or ""

if st.button("Analyze Resume"):

    if text.strip() == "":
        st.warning("Upload resume first")
    else:
        clean = clean_text(text)

        text_vec = vectorizer.transform([clean]).toarray()
        features = extract_features(clean)
        f_dict = extract_features_dict(clean)

        final_input = np.hstack((text_vec, [features]))

        proba = model.predict_proba(final_input)[0]
        fake_score = proba[1]

        result = 1 if fake_score > 0.65 else 0

        if result == 1:
            st.error(f"❌ Fake Resume ({fake_score*100:.2f}%)")
        else:
            st.success(f"✅ Real Resume ({(1-fake_score)*100:.2f}%)")

        st.progress(int(fake_score * 100))

        st.subheader("AI Explanation")
        if fake_score > 0.6:
            st.write("Model detected fake-like patterns")
        else:
            st.write("Resume looks consistent")

        st.subheader("Features")
        for k, v in f_dict.items():
            st.write(f"{k}: {v}")