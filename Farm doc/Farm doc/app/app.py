from flask import Flask, request, jsonify, render_template
import random
import numpy as np
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask_sqlalchemy import SQLAlchemy
import json

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///farmer.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db=SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Load metadata and model
lemmatizer = WordNetLemmatizer()
with open("words.pkl", "rb") as words_file:
    words = pickle.load(words_file)
with open("classes.pkl", "rb") as classes_file:
    classes = pickle.load(classes_file)

model = tf.keras.models.load_model("chatbot_model.h5")

with open("intents.json") as file:
    intents = json.load(file)

# Helper functions
def clean_up_sentence(sentence):
    tokens = word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])


# Route for the main page
@app.route("/")
def index():
    return render_template("Authentication.html")

# Route for Chatbot.html
@app.route("/Chatbot")
def chatbot():
    return render_template("Chatbot.html")

# Chatbot API route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    return jsonify({"response": response})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

