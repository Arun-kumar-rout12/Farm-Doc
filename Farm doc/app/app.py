from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import random
import numpy as np
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.preprocessing.image import img_to_array
import logging  
import json
import hashlib
import os
from werkzeug.utils import secure_filename
import pandas as pd
from PIL import Image

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///Farm.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = '1234SR34'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

class Farm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(64), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    farm_size = db.Column(db.Float, nullable=False)
    farming_experience = db.Column(db.Integer, nullable=False)
    farming_method = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<Farmer {self.name}>'

lemmatizer = WordNetLemmatizer()
with open("words.pkl", "rb") as words_file:
    words = pickle.load(words_file)
with open("classes.pkl", "rb") as classes_file:
    classes = pickle.load(classes_file)

model = tf.keras.models.load_model("chatbot_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.evaluate(np.zeros((1, len(words))), np.zeros((1, len(classes))))  

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
    if not sentence.strip():
        return [{"intent": "unknown", "probability": "0"}]
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return [{"intent": "unknown", "probability": "0"}]
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list or intents_list[0]["intent"] == "unknown":
        return "Sorry, I didn't understand that. Could you please rephrase?"

    tag = intents_list[0]["intent"]

    if tag == "problem_crop":
        crop_names = get_crop_names()
        if not crop_names:
            return "No crop data available."
        crop_options = "<h3>Select a Crop:</h3><ul>"
        for crop in crop_names:
            crop_options += f"<li><button onclick=\"fetchAndDisplayDiseases('{crop}')\">{crop}</button></li>"
        crop_options += "</ul>"
        return "Which crop are you having issues with?" + crop_options

    if tag == "problem_livestock":
        livestock_types = get_livestock_types()
        if not livestock_types:
            return "No livestock data available."
        livestock_options = "<h3>Select a Livestock Type:</h3><ul>"
        for livestock in livestock_types:
            livestock_options += f"<li><button onclick=\"fetchAndDisplayLivestockDiseases('{livestock}')\">{livestock}</button></li>"
        livestock_options += "</ul>"
        return "Which livestock are you having issues with?" + livestock_options

    if tag == "farming_advice":
        return "You can ask for general farming advice, and I'll provide tips and insights."

    if tag == "weather_update":
        return "Please visit the weather section for the latest updates."

    if tag == "loan_information":
        return "Visit the loan section for detailed information about farming loans."

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

dataset = pd.read_csv("crop_types_problems_solutions_seasons_months_updated.csv")

def get_crop_details(crop_name, crop_disease):
    crop_data = dataset[(dataset['Crop_Name'] == crop_name) & (dataset['Problem'] == crop_disease)]
    if not crop_data.empty:
        return crop_data.to_dict(orient='records')[0]
    return {}

def get_crop_names():
    if 'Crop_Name' not in dataset.columns:
        return []
    return dataset['Crop_Name'].unique().tolist()

livestock_dataset = pd.read_csv("Livestocks dataset.csv")

def get_livestock_types():
    if 'Livestock_Type' not in livestock_dataset.columns:
        return []
    return livestock_dataset['Livestock_Type'].unique().tolist()

def get_livestock_diseases(livestock_type):
    livestock_data = livestock_dataset[livestock_dataset['Livestock_Type'].str.lower() == livestock_type.lower()]
    if livestock_data.empty:
        return []
    return livestock_data['Disease'].unique().tolist()

def get_livestock_disease_details(livestock_type, disease_name):
    livestock_data = livestock_dataset[
        (livestock_dataset['Livestock_Type'].str.lower() == livestock_type.lower()) &
        (livestock_dataset['Disease'].str.lower() == disease_name.lower())
    ]
    if livestock_data.empty:
        return {}

    details = livestock_data.iloc[0].to_dict()
    symptoms_and_cures = {}
    for i in range(1, 7):  
        symptom = details.get(f"Symptom_{i}")
        cure = details.get(f"Cure_{i}")
        if symptom and cure:
            symptoms_and_cures[symptom] = cure

    details["Symptoms and Cures"] = symptoms_and_cures if symptoms_and_cures else {"Error": "No symptoms and cures available."}
    return details

@app.route("/")
def index():
    if 'authenticated' in session:
        return render_template("index.html")
    return redirect(url_for("Authentication"))

@app.route("/Authentication", methods=["GET", "POST"])
def Authentication():
    if request.method == "POST":
        name = request.form["farmerName"]
        password = request.form["password"]
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        farmer = Farm.query.filter_by(name=name).first()
        if farmer and farmer.password == hashed_password:
            session['authenticated'] = True
            session['farmerName'] = farmer.name
            return redirect(url_for("index"))
        return render_template("Authentication.html", error="Invalid Username and password.")
    return render_template("Authentication.html")

@app.route("/Logout")
def Logout():
    session.pop('authenticated', None)
    return redirect(url_for("Authentication"))

@app.route("/Register", methods=["GET", "POST"])
def Register():
    if request.method == "POST":
        data = request.form
        existing_farmer = Farm.query.filter_by(name=data["farmerName"]).first()
        
        if existing_farmer:
            return render_template("Register.html", error="Farmer with this username already exists!")

        try:
            hashed_password = hashlib.sha256(data["Password"].encode()).hexdigest()
            new_farmer = Farm(
                name=data["farmerName"],
                password=hashed_password,
                address=data["farmerAddress"],
                farm_size=float(data["farmSize"]),
                farming_experience=int(data["farmingExperience"]),
                farming_method=data["farmingMethod"]
            )
            db.session.add(new_farmer)
            db.session.commit()
            return redirect(url_for("Authentication"))
        except Exception as e:
            return render_template("Register.html", error=f"Error: {str(e)}")

    return render_template("Register.html")

@app.route("/Loan")
def Loan():
    return render_template("Loan.html")

@app.route("/hub")
def hub():
    return render_template("hub.html")

@app.route("/profile")
def profile():
    if 'authenticated' in session:
        name = session.get('farmerName')
        farmer = Farm.query.filter_by(name=name).first()
        if farmer:
            return render_template("Profile.html", farmer=farmer)
    return redirect(url_for("Authentication"))

@app.route("/crop")
def crop():
    return render_template("crops.html")

@app.route("/Wheather")
def Wheather():
    return render_template("Wheather.html")

@app.route("/Chatbot")
def chatbot():
    return render_template("Chatbot.html")

@app.route("/crop_details", methods=["POST"])
def crop_details():
    crop_name = request.form.get("crop_name")
    crop_disease = request.form.get("crop_disease")
    details = get_crop_details(crop_name, crop_disease)
    return jsonify(details)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.form
    message = data.get("message", "").strip()
    file = request.files.get("file")

    if not message and not file:
        return jsonify({"response": "<p>Please enter a message or upload a file.</p>"})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

    intents_list = predict_class(message)
    chatbot_response = get_response(intents_list, intents)

    if "Select a Crop" in chatbot_response:
        crop_names = get_crop_names()
        crop_options = "<h3>Select a Crop:</h3><ul>"
        for crop in crop_names:
            crop_options += f"<li><button onclick=\"fetchAndDisplayDiseases('{crop}')\">{crop}</button></li>"
        crop_options += "</ul>"
        chatbot_response += crop_options

    if "Select a Livestock Type" in chatbot_response:
        livestock_types = get_livestock_types()
        livestock_options = "<h3>Select a Livestock Type:</h3><ul>"
        for livestock in livestock_types:
            livestock_options += f"<li><button onclick=\"fetchAndDisplayLivestockDiseases('{livestock}')\">{livestock}</button></li>"
        livestock_options += "</ul>"
        chatbot_response += livestock_options

    return jsonify({"response": chatbot_response})

@app.route("/About_us")
def About_us():
    return render_template("About_us.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.form
    message = data.get('message')
    file = request.files.get('file')

    response = {}

    if message:
        intents_list = predict_class(message)
        response['message'] = get_response(intents_list, intents)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        response['file'] = f"Received file: {filename}"

    return jsonify(response)

@app.route('/get_crops', methods=['GET'])
def get_crops():
    if 'Crop_Name' not in dataset.columns:
        return jsonify([])
    crops = dataset['Crop_Name'].unique().tolist()
    return jsonify(crops)

@app.route("/get_diseases", methods=["GET"])
def get_diseases():
    crop_name = request.args.get("crop_name", "").strip().title()

    if not crop_name:
        return jsonify({"error": "Crop name is required."}), 400

    if "Crop_Name" not in dataset.columns or "Problem" not in dataset.columns:
        return jsonify({"error": "Dataset does not contain required columns."}), 500

    crop_data = dataset[dataset["Crop_Name"].str.lower() == crop_name.lower()]
    if crop_data.empty:
        return jsonify({"error": f"No data found for crop '{crop_name}'."}), 404

    diseases = crop_data["Problem"].unique().tolist()
    return jsonify(diseases)

@app.route("/crop_disease_details", methods=["GET"])
def crop_disease_details():
    crop_name = request.args.get("crop_name", "").strip().title()
    disease_name = request.args.get("disease_name", "").strip().replace("_", " ").title()  # Corrected line

    if not crop_name or not disease_name:
        return jsonify({"error": "Please provide both crop_name and disease_name"}), 400

    if "Crop_Name" not in dataset.columns or "Problem" not in dataset.columns:
        return jsonify({"error": "Dataset does not contain required columns"}), 500

    # Ensure consistent formatting for comparison
    crop_data = dataset[
        (dataset["Crop_Name"].str.strip().str.lower() == crop_name.lower()) & 
        (dataset["Problem"].str.strip().str.lower() == disease_name.lower())
    ]

    if crop_data.empty:
        return jsonify({"error": f"No details found for disease '{disease_name}' on crop '{crop_name}'"}), 404

    disease_details = crop_data.iloc[0].to_dict()
    
    response = {
        "Crop_Name": disease_details.get("Crop_Name", "N/A"),
        "Crop Type": disease_details.get("Crop_Type", "N/A"),
        "Farming Season": disease_details.get("Farming_Season", "N/A"),
        "Farming Month": disease_details.get("Farming_Month", "N/A"),
        "Problem": disease_details.get("Problem", "N/A"),
        "Solution": disease_details.get("Solution", "N/A")
    }

    return jsonify(response)

@app.route("/livestock_disease_details", methods=["GET"])
def livestock_disease_details_route():
    livestock_type = request.args.get("livestock_type", "").strip().title()
    disease_name = request.args.get("disease_name", "").strip().title()

    if not livestock_type or not disease_name:
        return jsonify({"error": "Please provide both livestock_type and disease_name"}), 400

    disease_details = get_livestock_disease_details(livestock_type, disease_name)
    if not disease_details:
        return jsonify({"error": f"No details found for disease '{disease_name}' in livestock type '{livestock_type}'"}), 404

    response = {
        "Livestock_Type": disease_details.get("Livestock_Type", "N/A"),
        "Disease": disease_details.get("Disease", "N/A"),
        "Symptoms and Cures": disease_details.get("Symptoms and Cures", {}),
    }

    return jsonify(response)

@app.route("/get_livestock_types", methods=["GET"])
def get_livestock_types_route():
    types = get_livestock_types()
    return jsonify(types)

@app.route("/get_livestock_diseases", methods=["GET"])
def get_livestock_diseases_route():
    livestock_type = request.args.get("livestock_type", "").strip().title()

    if not livestock_type:
        return jsonify({"error": "Livestock type is required."}), 400

    if "Livestock_Type" not in livestock_dataset.columns or "Disease" not in livestock_dataset.columns:
        return jsonify({"error": "Dataset does not contain required columns."}), 500

    diseases = get_livestock_diseases(livestock_type)
    if not diseases:
        return jsonify({"error": f"No data found for livestock type '{livestock_type}'."}), 404

    return jsonify(diseases)

@app.route("/livestock_details", methods=["POST"])
def livestock_details():
    livestock_type = request.form.get("livestock_type")
    disease_name = request.form.get("disease_name")
    details = get_livestock_disease_details(livestock_type, disease_name)
    return jsonify(details)

@app.route("/get_all_livestock_data", methods=["GET"])
def get_all_livestock_data():
    if "Livestock_Type" not in livestock_dataset.columns:
        return jsonify({"error": "Dataset does not contain required columns."}), 500

    livestock_data = livestock_dataset.to_dict(orient="records")
    return jsonify(livestock_data)

def load_model():
    model_path = "plant_disease.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the file exists.")
    return tf.keras.models.load_model(model_path)

def load_class_indices():
    with open("class_indices.json", "r") as file:
        class_indices = json.load(file)
    return {v: k for k, v in class_indices.items()}

image_model = load_model()
image_class_indices = load_class_indices()

def predict_disease(image):
    img = img_to_array(image)
    img = tf.image.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = image_model.predict(img)
    logging.debug(f"Raw predictions: {predictions}") 
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions)) 
    logging.debug(f"Predicted class: {predicted_class}, Confidence: {confidence}")
    logging.debug(f"Class probabilities: {dict(zip(image_class_indices.values(), predictions[0]))}")

    return image_class_indices.get(predicted_class, "Unknown"), confidence

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("file")
    message = request.form.get("message", "").strip()

    if not file and not message:
        return jsonify({"error": "Please provide a message or upload an image."}), 400

    response = {}

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            with Image.open(file_path).convert("RGB") as image:
                disease, confidence = predict_disease(image)
                response["image_prediction"] = {
                    "disease": disease,
                    "confidence": confidence 
                }
                logging.debug(f"Image prediction: {response['image_prediction']}")
        except Exception as e:
            response["image_error"] = f"Error processing image: {str(e)}"
            logging.error(f"Image processing error: {e}")

    if message:
        intents_list = predict_class(message)
        response["message_response"] = get_response(intents_list, intents)

    return jsonify(response)

@app.route("/handle_query", methods=["POST"])
def handle_query():
    query = request.form.get("query", "").lower()

    if "crop" in query:
        crop_name = request.args.get("crop_name", "").strip().title()
        if not crop_name:
            return jsonify({"error": "Crop name is required."}), 400

        if "Crop_Name" not in dataset.columns or "Problem" not in dataset.columns:
            return jsonify({"error": "Dataset does not contain required columns."}), 500

        crop_data = dataset[dataset["Crop_Name"].str.lower() == crop_name.lower()]
        if crop_data.empty:
            return jsonify({"error": f"No data found for crop '{crop_name}'."}), 404

        diseases = crop_data["Problem"].unique().tolist()
        return jsonify({"crop_name": crop_name, "diseases": diseases})

    elif "livestock" in query:
        livestock_type = request.args.get("livestock_type", "").strip().title()
        if not livestock_type:
            return jsonify({"error": "Livestock type is required."}), 400

        if "Livestock_Type" not in livestock_dataset.columns or "Disease" not in livestock_dataset.columns:
            return jsonify({"error": "Dataset does not contain required columns."}), 500

        diseases = get_livestock_diseases(livestock_type)
        if not diseases:
            return jsonify({"error": f"No data found for livestock type '{livestock_type}'."}), 404

        return jsonify({"livestock_type": livestock_type, "diseases": diseases})

    else:
        return jsonify({"error": "Query not recognized."}), 400

logging.basicConfig(level=logging.DEBUG) 

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    with app.app_context():
        db.create_all()
    app.run(debug=True)