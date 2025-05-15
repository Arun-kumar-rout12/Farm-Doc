import json
import random
import numpy as np
import tensorflow as tf
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

def load_data(file_path, default_value=[]):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return default_value

words = load_data("words.pkl")
classes = load_data("classes.pkl")

try:
    model = tf.keras.models.load_model("chatbot_model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open("intents.json") as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' not found.")
    intents = {"intents": []}

def clean_up_sentence(sentence):
    tokens = word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def bag_of_words(sentence):
    if not words:
        print("Warning: 'words.pkl' is empty. Bag-of-words cannot be generated.")
        return np.zeros(len(words))


    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

def predict_class(sentence):
    if model is None:
        return [{"intent": "error", "probability": "0.0"}]

    if not sentence.strip():
        return [{"intent": "error", "probability": "0.0"}]

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results] if results else []

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to help with that. Can you provide more details?"

    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    
    return "Sorry, I didn't understand that. Could you clarify?"

print("Chatbot is running! Type 'quit' to exit.")
while True:
    message = input("quit").strip()
    
    if message.lower() == "quit":
        print("Chatbot has stopped.")
        break

    intents_list = predict_class(message)
    response = get_response(intents_list, intents)

    print(f"Bot: {response}")