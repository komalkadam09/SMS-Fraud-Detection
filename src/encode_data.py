import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

# Remove unnecessary columns
data = data.drop(columns=[
    "Unnamed: 2",
    "Unnamed: 3",
    "Unnamed: 4"
])

# Rename columns
data.columns = ["label", "message"]

# Encode labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["message"])

print("TF-IDF Shape:")
print(X.shape)

print("\nEncoded Labels:")
print(data["label"].head())