import os
import pickle
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MAX_SEQUENCE_LENGTH = 100
emotion_categories = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'anger', 6:'love'}
emotion_names_list = [emotion_categories[i] for i in range(len(emotion_categories))]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'model.keras')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'tokenizer.pkl')

tokenizer = None
model = None

try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

try:
    model = load_model(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_text_for_model(text, tokenizer, max_sequence_length=100):
    if tokenizer is None:
        return None

    cleaned_text = text.lower()
    cleaned_text = re.sub(r"[^a-z0-9\s!?]", "", cleaned_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer failed to load. Check server logs."}), 500

    try:
        data = request.get_json()
        user_input = data.get('text', '')

        if not user_input:
            return jsonify({"error": "No 'text' provided in the request body."}), 400

        processed_input = preprocess_text_for_model(user_input, tokenizer, MAX_SEQUENCE_LENGTH)
        if processed_input is None:
            return jsonify({"error": "Text preprocessing failed."}), 500

        prediction_probabilities = model.predict(processed_input)
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotion_names_list, prediction_probabilities[0])}
        max_emotion = max(emotion_probs, key=emotion_probs.get)

        response_data = {
            'predicted_emotion': max_emotion,
            'probabilities': emotion_probs
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model_loaded": model is not None, "tokenizer_loaded": tokenizer is not None}), 200