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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Emotion Analyzer", layout="centered", initial_sidebar_state="collapsed")

# Clean, modern CSS styling - FIXED INPUT BOX STYLING
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .header h1 {
        font-size: 2.8rem;
        font-weight: 300;
        color: #2d3748;
        margin: 0 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .header p {
        font-size: 1.1rem;
        color: #718096;
        margin: 0;
        font-weight: 400;
    }
    
    /* Input section */
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .input-label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4a5568;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* FIXED: Streamlit text area styling - removed !important overrides that break functionality */
    .stTextArea textarea {
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif !important;
        background: #fafafa !important;
        transition: all 0.2s ease !important;
        resize: vertical !important;
        min-height: 120px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Hide the label that appears above */
    .stTextArea label {
        display: none !important;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 12px;
        transition: all 0.2s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
    }
    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .emotion-result {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .emotion-label {
        font-size: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: #718096;
        margin-top: 0.5rem;
    }
    
    /* Chart section */
    .chart-section {
        margin: 2rem 0;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* SHAP analysis */
    .shap-section {
        margin-top: 2rem;
    }
    
    .shap-text {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        font-size: 16px;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    /* Legend */
    .legend {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 12px;
        margin: 1.5rem 0;
        padding: 1.5rem;
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 500;
        color: #4a5568;
    }
    
    .legend-color {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* Hide Streamlit branding */
    .viewerBadge_container__1QSob {
        display: none;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2.2rem;
        }
        
        .input-section,
        .result-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .legend {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants
MAX_SEQUENCE_LENGTH = 100
emotion_categories = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'anger', 6:'love'}
emotion_names_list = [emotion_categories[i] for i in range(len(emotion_categories))]

TOKENIZER_FILE_ID = "19_8KtzNfKEyZJY3NsCtJyMbj4fAYxLrt"
MODEL_FILE_ID = "1E2sPDSR6m6vCFHut5tTXOswvjscfy81Q"
LOCAL_TOKENIZER_PATH = "tokenizer.pkl"
LOCAL_MODEL_PATH = "model.keras"

def preprocess_text_app(text, tokenizer, max_sequence_length=100):
    """Preprocess text for model prediction"""
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"[^a-z0-9\s!?]", "", cleaned_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

@st.cache_resource
def load_nlp_resources():
    """Load tokenizer and model from Google Drive"""
    creds = None
    try:
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(creds_info)
    except KeyError:
        st.error("GCP credentials not found. Please check your secrets configuration.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()

    service = None
    try:
        service = build('drive', 'v3', credentials=creds)
        
        # Download tokenizer if not exists
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

        # Download model if not exists
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
        st.error(f"Google Drive API error: {error}")
        st.stop()
    except Exception as e:
        st.error(f"Download error: {e}")
        st.stop()

    # Load tokenizer and model
    try:
        with open(LOCAL_TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        model = load_model(LOCAL_MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()

@st.cache_resource
def initialize_shap_explainer(_model, _tokenizer, background_size=50):
    """Initialize SHAP DeepExplainer with background data"""
    try:
        # Create diverse background samples for the explainer
        background_texts = [
            "I am happy today",
            "This is terrible news",
            "I feel scared about this",
            "What a wonderful surprise",
            "I love spending time with family",
            "This makes me angry",
            "The weather is nice",
            "I don't know how to feel",
            "Everything seems normal",
            "This is exciting news",
            "I feel sad about the situation",
            "This is frightening",
            "I am neutral on this topic",
            "What an amazing discovery",
            "I hate when this happens",
            "I adore this place",
            "This is confusing",
            "I feel peaceful",
            "This is disappointing",
            "I am thrilled about this opportunity"
        ]
        
        # Extend background samples if needed
        while len(background_texts) < background_size:
            background_texts.extend(background_texts[:min(len(background_texts), background_size - len(background_texts))])
        
        background_texts = background_texts[:background_size]
        
        # Preprocess background data
        background_data = []
        for text in background_texts:
            processed = preprocess_text_app(text, _tokenizer, MAX_SEQUENCE_LENGTH)
            background_data.append(processed[0])
        
        background_array = np.array(background_data)
        
        # Initialize SHAP DeepExplainer
        explainer = shap.DeepExplainer(_model, background_array)
        
        return explainer
        
    except Exception as e:
        st.error(f"Failed to initialize SHAP explainer: {e}")
        return None

def get_shap_explanations(text, explainer, tokenizer, emotion_names):
    """Get SHAP explanations for the input text"""
    try:
        # Preprocess the input text
        processed_input = preprocess_text_app(text, tokenizer, MAX_SEQUENCE_LENGTH)
        
        # Get SHAP values
        shap_values = explainer.shap_values(processed_input)
        
        # Convert to numpy array if needed
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        # Reshape if necessary - SHAP values should be (n_classes, n_samples, n_features)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, 0, :]  # Take first sample
        
        return shap_values, processed_input[0]
        
    except Exception as e:
        st.error(f"Error getting SHAP explanations: {e}")
        return None, None

def map_tokens_to_words(text, tokenized_sequence, tokenizer):
    """Map tokenized sequence back to original words"""
    try:
        words = text.split()
        
        # Get the reverse word index
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
        # Convert sequence to words
        sequence_words = []
        for token_id in tokenized_sequence:
            if token_id != 0:  # Skip padding tokens
                word = reverse_word_index.get(token_id, '<UNK>')
                sequence_words.append(word)
        
        # Try to align tokenized words with original words
        word_mappings = []
        seq_idx = 0
        
        for original_word in words:
            original_word_clean = re.sub(r"[^a-z0-9]", "", original_word.lower())
            
            # Find matching tokens
            token_indices = []
            temp_seq_idx = seq_idx
            
            # Look for exact match first
            if temp_seq_idx < len(sequence_words) and sequence_words[temp_seq_idx] == original_word_clean:
                token_indices.append(temp_seq_idx)
                seq_idx += 1
            else:
                # Look for partial matches or skip unknown words
                found = False
                for i in range(temp_seq_idx, min(temp_seq_idx + 3, len(sequence_words))):
                    if sequence_words[i] in original_word_clean or original_word_clean in sequence_words[i]:
                        token_indices.append(i)
                        seq_idx = i + 1
                        found = True
                        break
                
                if not found and seq_idx < len(sequence_words):
                    # Use the next available token
                    token_indices.append(seq_idx)
                    seq_idx += 1
            
            word_mappings.append({
                'word': original_word,
                'token_indices': token_indices
            })
        
        return word_mappings
        
    except Exception as e:
        print(f"Error in token mapping: {e}")
        return [{'word': word, 'token_indices': []} for word in text.split()]

def create_shap_colored_text_html(text, shap_values, tokenized_sequence, tokenizer, emotion_names):
    """Create HTML with color-coded words based on SHAP values"""
    try:
        words = text.split()
        word_mappings = map_tokens_to_words(text, tokenized_sequence, tokenizer)
        
        emotion_colors = {
            'joy': '#FFD700',
            'love': '#FF69B4', 
            'anger': '#FF4500',
            'fear': '#8A2BE2',
            'sadness': '#4169E1',
            'surprise': '#32CD32',
            'neutral': '#808080'
        }
        
        html_parts = []
        
        for word_info in word_mappings:
            word = word_info['word']
            token_indices = word_info['token_indices']
            
            if token_indices:
                # Aggregate SHAP values for this word across all classes
                word_shap_values = np.zeros(len(emotion_names))
                
                for token_idx in token_indices:
                    if token_idx < shap_values.shape[1]:  # Check bounds
                        # Sum across all emotion classes for this token
                        for class_idx in range(len(emotion_names)):
                            if class_idx < shap_values.shape[0]:
                                word_shap_values[class_idx] += shap_values[class_idx, token_idx]
                
                # Find the emotion with highest absolute SHAP value
                max_abs_idx = np.argmax(np.abs(word_shap_values))
                max_emotion = emotion_names[max_abs_idx]
                max_value = word_shap_values[max_abs_idx]
                
                # Only highlight if the SHAP value is significant
                if abs(max_value) > 0.01:  # Threshold for significance
                    color = emotion_colors.get(max_emotion, '#808080')
                    opacity = min(abs(max_value) * 2, 1.0)  # Scale opacity
                    
                    html_parts.append(
                        f'<span style="background-color: {color}; '
                        f'opacity: {opacity}; padding: 4px 8px; margin: 2px; '
                        f'border-radius: 6px; font-weight: 500; '
                        f'display: inline-block;" '
                        f'title="{max_emotion}: {max_value:.3f}">{word}</span>'
                    )
                else:
                    html_parts.append(f'<span style="margin: 2px; padding: 4px;">{word}</span>')
            else:
                # Word not found in tokenized sequence
                html_parts.append(f'<span style="margin: 2px; padding: 4px; opacity: 0.5;">{word}</span>')
        
        return ' '.join(html_parts)
        
    except Exception as e:
        st.error(f"Error creating colored text: {e}")
        return text

@st.cache_data
def analyze_with_shap(text, _explainer, _tokenizer, _emotion_names):
    """Analyze text with SHAP and return explanations"""
    try:
        if _explainer is None:
            return None, None
            
        shap_values, tokenized_sequence = get_shap_explanations(text, _explainer, _tokenizer, _emotion_names)
        return shap_values, tokenized_sequence
        
    except Exception as e:
        st.error(f"SHAP analysis failed: {e}")
        return None, None

# Load resources
tokenizer, model = load_nlp_resources()

# Initialize SHAP explainer
with st.spinner("Initializing AI explainer (this may take a moment)..."):
    explainer = initialize_shap_explainer(model, tokenizer)

# Header
st.markdown("""
<div class="header">
    <h1>Emotion Analyzer</h1>
    <p>AI-powered sentiment analysis with explainable results</p>
</div>
""", unsafe_allow_html=True)

# Single working input box
user_input = st.text_area(
    "Enter your text",
    height=120,
    placeholder="Write something to analyze its emotional content..."
)

analyze_button = st.button("Analyze", type="primary")

if analyze_button and user_input and user_input.strip():
    with st.spinner("Analyzing..."):
        time.sleep(0.3)
        
        processed_input = preprocess_text_app(user_input, tokenizer, MAX_SEQUENCE_LENGTH)
        
        try:
            prediction = model.predict(processed_input)
            emotion_probs = {emotion: round(float(prob), 3) for emotion, prob in zip(emotion_names_list, prediction[0])}
            max_emotion = max(emotion_probs, key=emotion_probs.get)
            
            # Main result
            st.markdown(f"""
            <div class="result-card">
                <div class="emotion-result">
                    <div class="emotion-label">{max_emotion}</div>
                    <div class="confidence-text">Confidence: {emotion_probs[max_emotion]:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Chart section
            st.markdown('<div class="result-card"><div class="section-title">Emotion Breakdown</div>', unsafe_allow_html=True)
            
            # Prepare data
            emotions = list(emotion_probs.keys())
            probabilities = list(emotion_probs.values())
            df = pd.DataFrame({'Emotion': emotions, 'Probability': probabilities}).sort_values('Probability', ascending=True)
            
            # Create clean chart
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('white')
            
            colors = ['#667eea' if emotion == max_emotion else '#e2e8f0' for emotion in df['Emotion']]
            bars = ax.barh(df['Emotion'], df['Probability'], color=colors, height=0.6)
            
            # Clean styling
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(0, max(probabilities) * 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, bottom=False)
            ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
            ax.set_facecolor('white')
            
            # Add value labels
            for bar, prob in zip(bars, df['Probability']):
                ax.text(prob + max(probabilities) * 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.2f}', ha='left', va='center', fontweight='500', fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # SHAP Analysis
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Word Analysis (SHAP)</div>', unsafe_allow_html=True)
            
            if explainer is not None:
                with st.spinner("Analyzing word contributions with SHAP..."):
                    shap_values, tokenized_sequence = analyze_with_shap(user_input, explainer, tokenizer, emotion_names_list)
                    
                    if shap_values is not None and tokenized_sequence is not None:
                        colored_html = create_shap_colored_text_html(
                            user_input, shap_values, tokenized_sequence, tokenizer, emotion_names_list
                        )
                        
                        st.markdown(f'<div class="shap-text">{colored_html}</div>', unsafe_allow_html=True)
                        
                        # Legend
                        st.markdown("""
                        <div class="legend">
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #FFD700;"></div>
                                <span>Joy</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #FF69B4;"></div>
                                <span>Love</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #FF4500;"></div>
                                <span>Anger</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #8A2BE2;"></div>
                                <span>Fear</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #4169E1;"></div>
                                <span>Sadness</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #32CD32;"></div>
                                <span>Surprise</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: #808080;"></div>
                                <span>Neutral</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <p style="font-size: 0.9rem; color: #718096; margin-top: 1rem; text-align: center;">
                            Words are colored based on their contribution to the predicted emotion. 
                            Hover over highlighted words to see their SHAP values.
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Could not generate SHAP explanations for this text.")
            else:
                st.warning("SHAP explainer not available. Word analysis unavailable.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")
            
elif analyze_button and not user_input.strip():
    st.warning("Please enter some text to analyze.")
