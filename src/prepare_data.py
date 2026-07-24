import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["message"])

# Target
y = data["label"]

print("Features Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)