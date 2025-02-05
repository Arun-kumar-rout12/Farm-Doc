import json
import random
import numpy as np
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load necessary files
lemmatizer = WordNetLemmatizer()
try:
    with open("words.pkl", "rb") as words_file:
        words = pickle.load(words_file)
    with open("classes.pkl", "rb") as classes_file:
        classes = pickle.load(classes_file)
    with open("disease_labels.pkl", "rb") as label_file:
        label_encoder = pickle.load(label_file)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Load models
chatbot_model = tf.keras.models.load_model("chatbot_model.h5")
disease_model = tf.keras.models.load_model("disease_model.h5")

# Load intents JSON
with open("intents.json") as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    tokens = word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = chatbot_model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

def predict_disease(symptom_values):
    symptom_array = np.array([symptom_values], dtype=float)
    predictions = disease_model.predict(symptom_array)[0]
    disease_index = np.argmax(predictions)
    disease_name = label_encoder[disease_index]
    return f"The most likely diagnosis is: {disease_name}. Apply appropriate treatment and ensure proper farm management."

print("Chatbot is running! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    
    # Check if input is symptom data (comma-separated 0s and 1s)
    if all(x in "01," for x in message.replace(" ", "")):
        try:
            symptom_values = list(map(int, message.split(",")))
            response = predict_disease(symptom_values)
        except:
            response = "Invalid input format. Please provide symptoms as comma-separated 0s and 1s."
    else:
        intents_list = predict_class(message)
        if intents_list:
            response = get_response(intents_list, intents)
        else:
            response = "I'm not sure how to help with that. Can you provide more details?"
    
    print(f"Bot: {response}")
