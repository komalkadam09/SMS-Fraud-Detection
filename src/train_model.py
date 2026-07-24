import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X = vectorizer.fit_transform(data["message"])

# Target
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("==============================")
print("Model Accuracy:")
print(accuracy)

print("\n==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n==============================")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully.")