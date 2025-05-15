import json
import random
import numpy as np
import tensorflow as tf
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

def load_pickle_file(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        exit()

base_dir = os.path.join("Farm doc")
words = load_pickle_file(os.path.join(base_dir, "words.pkl"))
classes = load_pickle_file(os.path.join(base_dir, "classes.pkl"))

try:
    with open(os.path.join(base_dir, "label_encoder.pkl"), "rb") as file:
        label_encoder = pickle.load(file)
except (EOFError, FileNotFoundError):
    print("Warning: Label encoder not found, continuing without it.")
    label_encoder = None

try:
    chatbot_model = tf.keras.models.load_model("chatbot_model.h5")
    chatbot_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Chatbot model loaded successfully.")
except Exception as e:
    print(f"Error loading chatbot model: {e}")
    exit()

try:
    with open(os.path.join(base_dir, "intents.json")) as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' not found.")
    exit()

def clean_up_sentence(sentence):
    tokens = word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def predict_class(sentence):
    if not sentence.strip():
        print("Empty input received.")
        return [{"intent": "unknown", "probability": "0"}]

    bow = bag_of_words(sentence)
    print(f"Bag of Words: {bow}")
    res = chatbot_model.predict(np.array([bow]))[0]
    print(f"Model Prediction: {res}")

    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        print("No intents matched the threshold.")
        return [{"intent": "unknown", "probability": "0"}]

    intents = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    print(f"Predicted Intents: {intents}")
    return intents

def get_response(intents_list, intents_json):
    if not intents_list or intents_list[0]["intent"] == "unknown":
        print("No valid intent found. Returning fallback response.")
        return "Sorry, I didn't understand that. Could you please rephrase?"

    tag = intents_list[0]["intent"]
    print(f"Matched Intent: {tag}")

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            print(f"Response: {response}")
            return response

    print("No matching response found in intents.json.")
    return "Sorry, I don't have an answer for that."

def train_model():
    try:
        with open(os.path.join(base_dir, "intents.json")) as file:
            intents = json.load(file)
    except FileNotFoundError:
        print("Error: 'intents.json' not found.")
        return

    training_sentences = []
    training_labels = []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = word_tokenize(pattern)
            training_sentences.append([lemmatizer.lemmatize(word.lower()) for word in tokens])
            training_labels.append(intent["tag"])

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(training_labels)

    with open(os.path.join(base_dir, "label_encoder.pkl"), "wb") as file:
        pickle.dump(label_encoder, file)

    training_data = []
    for sentence in training_sentences:
        bow = [1 if w in sentence else 0 for w in words]
        training_data.append(bow)

    training_data = np.array(training_data)
    labels = np.array(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(words),), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(training_data, labels, epochs=200, batch_size=5, verbose=1)

    model.save("chatbot_model.h5")
    print("Model trained and saved as 'chatbot_model.h5'")

print("Chatbot is running! Type 'quit' to exit.")
try:
    while True:
        message = input("You: ").strip()
        
        if message.lower() == "quit":
            print("Chatbot session ended. Goodbye!")
            break

        if re.fullmatch(r"([01],?)+", message):
            try:
                symptom_values = list(map(int, message.split(",")))
                response = "Symptom data received. Processing..."
            except ValueError:
                response = "Invalid input format. Please provide symptoms as comma-separated 0s and 1s."
        else:
            intents_list = predict_class(message)
            response = get_response(intents_list, intents)
        
        print(f"Bot: {response}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Chatbot is shutting down.")