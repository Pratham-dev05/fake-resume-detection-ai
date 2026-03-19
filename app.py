import streamlit as st
import pickle
import numpy as np
from PyPDF2 import PdfReader

from src.preprocessing import clean_text
from src.features import extract_features, extract_features_dict

# Load
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake Resume Detection", layout="centered")

st.markdown("<h1 style='text-align:center;'>🤖 Fake Resume Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered Resume Authenticity Analyzer</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

# Explanation
def explain_resume(text, f, fake_score):
    reasons = []
    t = text.lower()

    if fake_score > 0.6:
        reasons.append("Model detected patterns similar to fake resumes")

    if f["Skill Count"] > 10 and f["Experience Years"] < 2:
        reasons.append("High skills but low experience — possible exaggeration")

    if f["Word Count"] < 30:
        reasons.append("Resume too short and lacks detail")

    if f["Repetition Score"] > 25:
        reasons.append("Keyword repetition detected")

    if "expert" in t and "month" in t:
        reasons.append("Expert claim with very low experience")

    if len(reasons) == 0:
        reasons.append("Resume appears consistent and realistic")

    return reasons


if st.button("🔍 Analyze Resume"):

    if text.strip() == "":
        st.warning("⚠️ Upload resume first")
    else:
        clean = clean_text(text)

        text_vec = vectorizer.transform([clean]).toarray()
        features = extract_features(clean)
        f_dict = extract_features_dict(clean)

        final_input = np.hstack((text_vec, [features]))

        proba = model.predict_proba(final_input)[0]
        fake_score = proba[1]

        # 🔥 STABLE THRESHOLD
        result = 1 if fake_score > 0.65 else 0

        st.subheader("📊 Result")

        if result == 1:
            st.error(f"❌ Fake Resume ({fake_score*100:.2f}%)")
        else:
            st.success(f"✅ Real Resume ({(1-fake_score)*100:.2f}%)")

        # Progress bar
        st.subheader("📊 Fake Probability")
        st.progress(int(fake_score * 100))

        # Quality
        st.subheader("⭐ Resume Quality Score")
        st.write(f"{(1-fake_score)*100:.2f}/100")

        # Explanation
        st.subheader("🧠 AI Explanation")
        reasons = explain_resume(text, f_dict, fake_score)
        for r in reasons:
            st.write("•", r)

        # Features
        st.subheader("⚙️ Features")
        for k, v in f_dict.items():
            st.write(f"{k}: {v}")

        # Preview
        st.subheader("📄 Resume Preview")
        st.write(text[:500])