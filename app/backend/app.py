import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as grequests
except ImportError:
    id_token = None
    grequests = None

from predict import predict_skin_lesion

app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")


def _get_collections():
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed. Install pymongo to use auth/chat persistence.")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client["ai_menthor"]
    return db["users"], db["chats"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "medmentor-inference-api"})


# -------- Auth --------
@app.route("/signup", methods=["POST"])
def signup():
    users_col, _ = _get_collections()
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "Missing username or password"}), 400

    if users_col.find_one({"username": username}):
        return jsonify({"success": False, "message": "User already exists"}), 400

    hash_pw = generate_password_hash(password)
    users_col.insert_one({"username": username, "password": hash_pw})
    return jsonify({"success": True})


@app.route("/login", methods=["POST"])
def login():
    users_col, _ = _get_collections()
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "Missing username or password"}), 400

    user = users_col.find_one({"username": username})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return jsonify({"success": True, "username": username})


# -------- Google OAuth2 --------
@app.route("/google-login", methods=["POST"])
def google_login():
    if id_token is None or grequests is None:
        return jsonify({"success": False, "message": "google-auth is not installed"}), 500
    try:
        users_col, _ = _get_collections()
        token = request.json.get("token")
        idinfo = id_token.verify_oauth2_token(token, grequests.Request(), GOOGLE_CLIENT_ID)

        email = idinfo["email"]
        name = idinfo.get("name", email.split("@")[0])

        user = users_col.find_one({"username": email})
        if not user:
            users_col.insert_one({"username": email, "name": name, "google": True})

        return jsonify({"success": True, "username": name})
    except ValueError:
        return jsonify({"success": False, "message": "Invalid Google token"}), 400


# -------- Chats --------
@app.route("/save-chat", methods=["POST"])
def save_chat():
    _, chats_col = _get_collections()
    data = request.get_json(force=True)
    chats_col.update_one({"id": data["id"]}, {"$set": data}, upsert=True)
    return jsonify({"success": True})


@app.route("/get-chats", methods=["GET"])
def get_chats():
    _, chats_col = _get_collections()
    chats = list(chats_col.find({}, {"_id": 0}))
    return jsonify(chats)


# -------- Predict --------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(path)

    try:
        result = predict_skin_lesion(path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
