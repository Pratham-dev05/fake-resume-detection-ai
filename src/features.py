import re

def extract_experience(text):
    match = re.search(r'(\d+)\s+year', text)
    if match:
        return int(match.group(1))
    return 0


def extract_features_dict(text):
    words = text.split()

    features = {}

    features['Resume Length'] = len(text)
    features['Word Count'] = len(words)
    features['Unique Words'] = len(set(words))
    features['Repetition Score'] = len(words) - len(set(words))

    skills = ["python", "java", "ai", "ml", "sql", "react", "node", "cloud", "aws"]
    features['Skill Count'] = sum([text.lower().count(skill) for skill in skills])

    features['Experience Years'] = extract_experience(text.lower())

    return features


def extract_features(text):
    return list(extract_features_dict(text).values())