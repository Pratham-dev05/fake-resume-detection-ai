import streamlit as st
import pickle
import numpy as np
import os
from PyPDF2 import PdfReader
from sklearn.exceptions import NotFittedError

from src.preprocessing import clean_text
from src.features import extract_features, extract_features_dict

# 🔥 BASE PATH FIX
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model & vectorizer safely
try:
    model = pickle.load(open(os.path.join(BASE_DIR, "models/model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "models/vectorizer.pkl"), "rb"))
except Exception as e:
    st.error("❌ Model files not found or corrupted. Please retrain and upload.")
    st.stop()

# UI Setup
st.set_page_config(page_title="Fake Resume Detection", layout="centered")

st.markdown("<h1 style='text-align:center;'>🤖 Fake Resume Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered Resume Authenticity Analyzer</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

text = ""

# Extract text from PDF
if uploaded_file:
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t
    except:
        st.error("❌ Error reading PDF file")
        st.stop()

# 🔥 AI Explanation Logic (Improved)
def explain_resume(text, f, fake_score):
    reasons = []
    t = text.lower()

    if fake_score > 0.6:
        reasons.append("Model detected patterns similar to fake or exaggerated resumes")

    if f["Skill Count"] > 10 and f["Experience Years"] < 2:
        reasons.append("Too many skills listed compared to low experience")

    if f["Word Count"] < 40:
        reasons.append("Resume is too short and lacks detailed information")

    if f["Repetition Score"] > 30:
        reasons.append("High repetition of keywords detected (possible keyword stuffing)")

    if "expert" in t and "month" in t:
        reasons.append("Unrealistic claim: expert level in very short time")

    if "google" in t and "microsoft" in t:
        reasons.append("Multiple top companies mentioned — verify authenticity")

    if len(reasons) == 0:
        reasons.append("Resume content appears consistent and realistic")

    return reasons

# 🔍 ANALYSIS BUTTON
if st.button("🔍 Analyze Resume"):

    if text.strip() == "":
        st.warning("⚠️ Please upload a resume first")
    else:
        clean = clean_text(text)

        try:
            # Vectorization
            text_vec = vectorizer.transform([clean]).toarray()
        except NotFittedError:
            st.error("❌ Model not trained properly. Please retrain the model.")
            st.stop()

        # Features
        features = extract_features(clean)
        f_dict = extract_features_dict(clean)

        # Combine
        final_input = np.hstack((text_vec, [features]))

        # Prediction
        proba = model.predict_proba(final_input)[0]
        fake_score = proba[1]

        # 🔥 Stable Threshold
        result = 1 if fake_score > 0.65 else 0

        # RESULT
        st.subheader("📊 Result")

        if result == 1:
            st.error(f"❌ Fake Resume ({fake_score*100:.2f}% confidence)")
        else:
            st.success(f"✅ Real Resume ({(1-fake_score)*100:.2f}% confidence)")

        # Progress Bar
        st.subheader("📊 Fake Probability")
        st.progress(int(fake_score * 100))

        # Quality Score
        st.subheader("⭐ Resume Quality Score")
        st.write(f"{(1-fake_score)*100:.2f}/100")

        # Explanation
        st.subheader("🧠 AI Explanation")
        reasons = explain_resume(text, f_dict, fake_score)
        for r in reasons:
            st.write("•", r)

        # Features
        st.subheader("⚙️ Extracted Features")
        for k, v in f_dict.items():
            st.write(f"{k}: {v}")

        # Download Report
        report = f"""
Result: {'Fake' if result==1 else 'Real'}
Confidence: {fake_score*100:.2f}%

Reasons:
{chr(10).join(reasons)}
"""
        st.download_button("📄 Download Report", report)

        # Preview
        st.subheader("📄 Resume Preview")
        st.write(text[:500])