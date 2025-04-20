import joblib

# Load saved model and vectorizer
model = joblib.load("RandomForest_best_model.pkl")
vectorizer = joblib.load("RandomForest_vectorizer.pkl")

print("=== NuSVC Sentiment Classifier ===")
print("Type a sentence to classify (or type 'exit' to quit):\n")

while True:
    sentence = input("Your sentence: \n").strip()
    
    if sentence.lower() == "exit":
        print("Goodbye!")
        break

    if not sentence:
        print("Please enter a valid sentence.\n")
        continue

    # Transform and predict
    X_input = vectorizer.transform([sentence])
    prediction = model.predict(X_input)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
    print(f"Prediction: {sentiment}\n")
