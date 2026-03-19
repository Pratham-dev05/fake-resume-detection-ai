import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    words = text.lower().split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)