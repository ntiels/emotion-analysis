import streamlit as st
import pickle
import re
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import streamlit.components.v1 as components

st.set_page_config(page_title="NLP Emotion Analyzer", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .input-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .custom-textbox, .stTextArea textarea {
        width: 100%;
        min-height: 120px !important;
        padding: 15px !important;
        border: 2px solid #ddd !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-family: 'Arial', sans-serif !important;
        resize: vertical;
        transition: border-color 0.3s ease;
        box-sizing: border-box;
    }
    
    .custom-textbox:focus, .stTextArea textarea:focus {
        outline: none !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 150px;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    }
    
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .emotion-result {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .emotion-scores {
        font-family: 'Courier New', monospace;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #eee;
    }
    
    .stSpinner {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

MAX_SEQUENCE_LENGTH = 100
emotion_categories = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'anger', 6:'love'}
emotion_names_list = [emotion_categories[i] for i in range(len(emotion_categories))]

TOKENIZER_FILE_ID = "19_8KtzNfKEyZJY3NsCtJyMbj4fAYxLrt"
MODEL_FILE_ID = "1E2sPDSR6m6vCFHut5tTXOswvjscfy81Q"
LOCAL_TOKENIZER_PATH = "tokenizer.pkl"
LOCAL_MODEL_PATH = "model.keras"


@st.cache_resource
def load_nlp_resources():
    creds = None
    try:
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_info)
    except KeyError:
        st.error("GCP credentials ('gcp_service_account') not found in Streamlit secrets. Please check your secrets.toml file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading GCP credentials: {e}")
        st.stop()

    service = None
    try:
        service = build('drive', 'v3', credentials=creds)

        if not os.path.exists(LOCAL_TOKENIZER_PATH):
            request = service.files().get_media(fileId=TOKENIZER_FILE_ID)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            with open(LOCAL_TOKENIZER_PATH, 'wb') as f:
                f.write(fh.read())

        if not os.path.exists(LOCAL_MODEL_PATH):
            request = service.files().get_media(fileId=MODEL_FILE_ID)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                f.write(fh.read())
    except HttpError as error:
        st.error(f"An error occurred with Google Drive API: {error}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        st.stop()

    try:
        with open(LOCAL_TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        model = load_model(LOCAL_MODEL_PATH)
        return tokenizer, model
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

# Custom HTML header
st.markdown("""
<div class="main-header">
    <h1>🎭 Sentiment Analysis</h1>
    <p>BiGRU Neural Network with GloVe Embeddings</p>
</div>
""", unsafe_allow_html=True)

# Custom HTML styling for Streamlit components
st.markdown("""
<div class="input-container">
    <h3 style="margin-top: 0; color: #333;">Enter your text for emotion analysis:</h3>
</div>
""", unsafe_allow_html=True)

# Enhanced Streamlit text area with custom styling
user_input = st.text_area(
    "",
    height=120,
    placeholder="Type something like 'This movie was fantastic!' or 'I hated the food.'",
    help="Enter any text to analyze its emotional sentiment"
)
# Custom styled button using Streamlit
analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if analyze_button and user_input and user_input.strip():
    with st.spinner("🔄 Analyzing your text..."):
        time.sleep(0.5)  # Small delay for better UX
        processed_input = preprocess_text_app(user_input, tokenizer, MAX_SEQUENCE_LENGTH)

        try:
            prediction = model.predict(processed_input)
            emotion_probs = {emotion: round(float(prob), 3) for emotion, prob in zip(emotion_names_list, prediction[0])}
            max_emotion = max(emotion_probs, key=emotion_probs.get)
            
            # Custom HTML results display
            st.markdown(f"""
            <div class="result-container">
                <div class="emotion-result">
                    🎯 <strong>Predicted Emotion:</strong> <span style="color: #667eea; font-size: 1.3rem;">{max_emotion.upper()}</span>
                </div>
                <div>
                    <strong>📊 Confidence Scores:</strong>
                    <div class="emotion-scores">
                        {emotion_probs}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.warning("Please ensure your model's input shape and prediction logic match your training setup.")

elif analyze_button and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")

# Fallback: Original Streamlit input (for backup)
st.markdown("---")
st.markdown("### Alternative Input (Streamlit Native)")
fallback_input = st.text_area("Enter text here (fallback):", height=100, placeholder="Backup input method")

if st.button("🔍 Analyze (Fallback)"):
    if fallback_input:
        with st.spinner("Analyzing..."):
            processed_input = preprocess_text_app(fallback_input, tokenizer, MAX_SEQUENCE_LENGTH)

            try:
                prediction = model.predict(processed_input)
                emotion_probs = {emotion: round(float(prob), 3) for emotion, prob in zip(emotion_names_list, prediction[0])}
                max_emotion = max(emotion_probs, key=emotion_probs.get)
                
                st.success(f"**Predicted emotion:** `{max_emotion}`")
                st.info(f"**Prediction Scores:** `{emotion_probs}`")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.warning("Please ensure your model's input shape and prediction logic match your training setup.")
    else:
        st.warning("Please enter some text to analyze.")

# Custom footer
st.markdown("""
<div class="footer">
    <hr style="margin: 2rem 0; border: none; height: 1px; background: #eee;">
    <p>💻 <strong>Developed by Nathaniel Shin</strong></p>
    <p style="font-size: 0.9rem; color: #888;">
        Powered by BiGRU Neural Network & GloVe Word Embeddings
    </p>
</div>
""", unsafe_allow_html=True)
