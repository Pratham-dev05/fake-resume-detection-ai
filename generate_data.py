import pandas as pd
import random

# 🔥 FIX RANDOMNESS
random.seed(42)

skills = ["Python", "Java", "AI", "ML", "SQL", "React", "Node", "Cloud", "AWS", "Docker"]

real_templates = [
    "Software Engineer with {} years experience in {} and {}. Worked on real-world projects and APIs.",
    "Developed applications using {} and {} for {} years with strong problem solving.",
    "Backend developer with {} years experience using {} and databases like {}.",
    "Worked on machine learning projects using {} and {} with {} years experience.",
    "Frontend developer skilled in {} and {} with {} years of experience."
]

fake_templates = [
    "Expert in {} {} {} {} {} with 1 month experience",
    "Master in {} {} {} {} {} {} in 2 weeks",
    "Knows {} {} {} {} {} {} {} with no real experience",
    "Fresher but expert in {} {} {} {} {}",
    "Worked at Google Microsoft Amazon simultaneously for 1 year"
]

data = []

# Real (5000)
for _ in range(5000):
    text = random.choice(real_templates).format(
        random.randint(1, 5),
        random.choice(skills),
        random.choice(skills)
    )
    data.append([text, 0])

# Fake (5000)
for _ in range(5000):
    text = random.choice(fake_templates).format(
        random.choice(skills),
        random.choice(skills),
        random.choice(skills),
        random.choice(skills),
        random.choice(skills),
        random.choice(skills),
        random.choice(skills)
    )
    data.append([text, 1])

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/resumes.csv", index=False)

print("🔥 Stable 10,000 dataset created!")