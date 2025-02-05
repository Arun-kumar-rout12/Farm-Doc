import json
import random
import numpy as np
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from chatbot import predict_class, predict_disease

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load words and classes with error handling
def load_data(file_path, default_value=[]):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return default_value

words = load_data("words.pkl")
classes = load_data("classes.pkl")

# Load the trained model with error handling
try:
    model = tf.keras.models.load_model("chatbot_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load intents file
try:
    with open("intents.json") as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' not found.")
    intents = []

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence"""
    tokens = word_tokenize(sentence)  # Tokenizing the sentence
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]  # Lemmatizing each token

def bag_of_words(sentence):
    """Convert sentence to a bag of words vector"""
    sentence_words = clean_up_sentence(sentence)  # Clean and tokenize the sentence
    return np.array([1 if w in sentence_words else 0 for w in words])  # Create a binary vector based on words

def predict_class(sentence):
    """Predict the intent class of a given sentence"""
    if model is None:
        return [{"intent": "error", "probability": "0.0"}]
    
    bow = bag_of_words(sentence)  # Convert sentence to bag of words
    res = model.predict(np.array([bow]))[0]  # Make prediction using the trained model
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """Get the chatbot's response based on predicted intent"""
    tag = intents_list[0]["intent"]
    
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])  # Pick a random response from the matched intent
            # Optionally, add some randomness for varied responses
            return response
    return "Sorry, I didn't understand that. Could you clarify?"

# Main chat loop
print("Chatbot is running! Type 'quit' to exit.")
while True:
    message = input("You: ")
    
    if message.lower() == "quit":
        break
    
    # Predict the intent
    intents_list = predict_class(message)
    
    # Get the response based
