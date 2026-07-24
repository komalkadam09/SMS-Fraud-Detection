import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Take user input
message = input("Enter SMS: ")

# Convert text to TF-IDF
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: HAM (Legitimate Message)")