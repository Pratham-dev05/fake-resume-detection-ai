# 🤖 Fake Resume Detection using AI & Machine Learning

## 📌 Project Description

This project presents an AI-powered system designed to detect fake resumes using advanced Natural Language Processing (NLP) and Machine Learning techniques. 

The system analyzes resume content, extracts meaningful features such as skill count, experience consistency, keyword density, and repetition patterns, and classifies resumes as **Real or Fake** using a trained Random Forest model.

To ensure robustness and accuracy, the model is trained on a large dataset of **10,000+ resumes** (both real and synthetically generated fake resumes). The system also includes an **Explainable AI module**, which provides clear reasons for each prediction, making it transparent and interpretable.

Additionally, the application features a user-friendly interface built with Streamlit, allowing users to upload PDF resumes and receive instant analysis along with confidence scores and quality evaluation.

---

## 🚀 Key Highlights

- 📄 PDF Resume Upload & Analysis  
- 🧠 NLP-based Text Processing  
- 📊 TF-IDF + Feature Engineering  
- 🤖 Random Forest Machine Learning Model  
- 📈 Accuracy ~90%  
- 🧾 Explainable AI (Reason for prediction)  
- ⭐ Resume Quality Score  
- 📊 Confusion Matrix & Performance Graphs  
- 📦 Large Dataset (10,000+ samples)  

---

## 🎯 Objective

The main goal of this project is to automate resume verification and assist recruiters in identifying fraudulent or exaggerated resumes efficiently.

---

## 🧠 Technologies Used

- Python  
- Scikit-learn  
- NLTK  
- Streamlit  
- Pandas  

---

## ▶️ How to Run

```bash
python generate_data.py
cd src
python train.py
cd ..
streamlit run app.py