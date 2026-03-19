import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from preprocessing import clean_text
from features import extract_features

# Load dataset
df = pd.read_csv("../data/resumes.csv")

texts = df["text"]
labels = df["label"]

# Clean text
cleaned = [clean_text(t) for t in texts]

# 🔥 FIXED VECTORIZER
vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
X_text = vectorizer.fit_transform(cleaned).toarray()

# Features
X_features = [extract_features(t) for t in cleaned]

# Combine
X = np.hstack((X_text, X_features))

# 🔥 FIXED SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# 🔥 FIXED MODEL
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("🔥 Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center')

plt.savefig("../models/confusion_matrix.png")
plt.close()

# Save
pickle.dump(model, open("../models/model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))

print("✅ Stable model trained!")