# routes.py
from app import app, db
from app.models import User
from flask import request, jsonify

@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    new_user = User(username=data['username'], Password=data['Password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User added successfully!"})

@app.route('/get_users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{"username": user.username, "email": user.email} for user in users])