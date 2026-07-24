import joblib

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("===================================")
print("      SMS SPAM DETECTION SYSTEM")
print("===================================")

# Take SMS from user
message = input("\nEnter your SMS: ")

# Convert message into TF-IDF features
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("🚫 This SMS is SPAM.")
else:
    print("✅ This SMS is HAM (Legitimate).")