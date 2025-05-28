import streamlit as st
import pickle
import re
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

st.set_page_config(page_title="NLP Emotion Analyzer", layout="centered")

MAX_SEQUENCE_LENGTH = 100
emotion_categories = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'anger', 6:'love'}
emotion_names_list = [emotion_categories[i] for i in range(len(emotion_categories))]

# Google Drive file IDs (replace with your actual file IDs)
TOKENIZER_FILE_ID = "19_8KtzNfKEyZJY3NsCtJyMbj4fAYxLrt"
MODEL_FILE_ID = "1E2sPDSR6m6vCFHut5tTXOswvjscfy81Q"
LOCAL_TOKENIZER_PATH = "tokenizer.pkl"
LOCAL_MODEL_PATH = "model.keras"


@st.cache_resource
def load_nlp_resources():
    """
    Downloads the tokenizer and model from Google Drive, and then loads them.
    """

    creds_json_string = st.secrets["gcp_service_account"]
    creds_info = json.loads(creds_json_string)

    creds = service_account.Credentials.from_service_account_info(creds_info)

    try:
        service = build('drive', 'v3', credentials=creds)
        if not os.path.exists(LOCAL_TOKENIZER_PATH):
            st.info(f"Downloading tokenizer from Google Drive...")
            request = service.files().get_media(fileId=TOKENIZER_FILE_ID)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            with open(LOCAL_TOKENIZER_PATH, 'wb') as f:
                f.write(fh.read())
            st.success("Tokenizer downloaded successfully.")

        if not os.path.exists(LOCAL_MODEL_PATH):
            st.info(f"Downloading model from Google Drive...")
            request = service.files().get_media(fileId=MODEL_FILE_ID)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                f.write(fh.read())
            st.success("Model downloaded successfully.")

    except HttpError as error:
        st.error(f"An error occurred: {error}")
        st.stop()
    except Exception as e:
         st.error(f"An unexpected error occurred: {e}")
         st.stop()

    try:
        with open(LOCAL_TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        model = load_model(LOCAL_MODEL_PATH)
        return tokenizer, model
    except FileNotFoundError:
        st.error("Error: tokenizer.pkl or model.keras not found. Please ensure they were downloaded correctly.")
        st.stop()
    except pickle.UnpicklingError:
        st.error("Error: Could not load tokenizer. Ensure it's a valid pickle file.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading tokenizer or model: {e}")
        st.stop()



tokenizer, model = load_nlp_resources()

def preprocess_text_app(text, tokenizer, max_sequence_length=100):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"[^a-z0-9\s!?]", "", cleaned_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# app layout

st.title("Sentiment Analysis with BiGRU and GloVe Embeddings")
st.markdown("""
    This app predicts the emotion of text using a BiGRU neural network
    trained with GloVe pre-trained word embeddings.
    """)

user_input = st.text_area("Enter text here:", height=150, placeholder="Type something like 'This movie was fantastic!' or 'I hated the food.'")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing..."):
            processed_input = preprocess_text_app(user_input, tokenizer, MAX_SEQUENCE_LENGTH)

            try:
                prediction = model.predict(processed_input)
                emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotion_names_list, prediction[0])}
                max_emotion = max(emotion_probs, key=emotion_probs.get)
                output = {
                    'probabilities': emotion_probs,
                    'predicted_emotion': max_emotion
                }
                st.info(f"Predicted emotion: `{max_emotion}`")
                st.info(f"Prediction Scores: `{output}`")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.warning("Please ensure your model's input shape and prediction logic match your training setup.")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("Developed by Nathaniel Shin")
st.markdown("This app is for demonstration purposes. Model accuracy may vary.")
