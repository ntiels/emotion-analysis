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
    
    .shap-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .shap-text {
        font-family: 'Courier New', monospace;
        background: white;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #e9ecef;
        line-height: 1.6;
    }
    
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 5px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 3px;
        border: 1px solid #ccc;
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
        st.error("GCP credentials ('gcp_service_account') not found in Streamlit secrets. Please check your secrets.toml file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading GCP credentials: {e}")
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
        st.error(f"An error occurred with Google Drive API: {error}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        st.stop()

    # Load tokenizer and model
    try:
        with open(LOCAL_TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        model = load_model(LOCAL_MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"An unexpected error occurred loading tokenizer or model: {e}")
        st.stop()

def create_shap_explainer(model, tokenizer, max_length=100):
    """Create SHAP explainer for the model"""
    def model_predict(texts):
        # Handle both single strings and lists of strings
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # Preprocess texts same way as training
        processed_texts = []
        for text in texts:
            if isinstance(text, (list, np.ndarray)):
                text = str(text)
            cleaned_text = str(text).lower()
            cleaned_text = re.sub(r"[^a-z0-9\s!?]", "", cleaned_text)
            processed_texts.append(cleaned_text)
        
        # Convert to sequences and pad
        sequences = tokenizer.texts_to_sequences(processed_texts)
        padded = pad_sequences(sequences, maxlen=max_length)
        
        # Get predictions
        predictions = model.predict(padded, verbose=0)
        return predictions
    
    # Create explainer - use empty strings as background
    background_data = [""] * 5
    explainer = shap.Explainer(model_predict, background_data)
    return explainer

def get_word_level_shap_values(text, model, tokenizer, max_length=100):
    """Get SHAP values for individual words in the text"""
    try:
        words = text.split()
        if len(words) == 0:
            return None
            
        # Create word-level explanations by masking individual words
        word_contributions = np.zeros((len(words), len(emotion_names_list)))
        
        # Get baseline prediction (empty text)
        baseline_input = preprocess_text_app("", tokenizer, max_length)
        baseline_pred = model.predict(baseline_input, verbose=0)[0]
        
        # Get full text prediction
        full_input = preprocess_text_app(text, tokenizer, max_length)
        full_pred = model.predict(full_input, verbose=0)[0]
        
        # Calculate contribution of each word by removing it
        for i, word in enumerate(words):
            # Create text without this word
            masked_words = words[:i] + words[i+1:]
            masked_text = " ".join(masked_words)
            
            if masked_text.strip():
                masked_input = preprocess_text_app(masked_text, tokenizer, max_length)
                masked_pred = model.predict(masked_input, verbose=0)[0]
            else:
                masked_pred = baseline_pred
            
            # Contribution = full_prediction - prediction_without_word
            word_contributions[i] = full_pred - masked_pred
        
        return word_contributions
        
    except Exception as e:
        print(f"Error in word-level SHAP: {e}")
        return None

def create_colored_text_html(text, shap_values, emotion_names, predicted_emotion):
    """Create HTML with color-coded words based on SHAP values"""
    words = text.split()
    
    # Color scheme for different emotions
    emotion_colors = {
        'joy': '#FFD700',      # Gold
        'love': '#FF69B4',     # Hot Pink
        'anger': '#FF4500',    # Red Orange
        'fear': '#8A2BE2',     # Blue Violet
        'sadness': '#4169E1',  # Royal Blue
        'surprise': '#32CD32', # Lime Green
        'neutral': '#808080'   # Gray
    }
    
    html_parts = []
    
    for i, word in enumerate(words):
        if i < len(shap_values):
            # Find the emotion with highest contribution for this word
            word_contribs = shap_values[i]
            max_contrib_idx = np.argmax(np.abs(word_contribs))
            max_emotion = emotion_names[max_contrib_idx]
            contribution = word_contribs[max_contrib_idx]
            
            # Only color if contribution is significant
            if abs(contribution) > 0.05:  # Threshold for highlighting
                color = emotion_colors.get(max_emotion, '#808080')
                opacity = min(abs(contribution) * 10, 1.0)  # Scale opacity based on contribution
                
                html_parts.append(
                    f'<span style="background-color: {color}; '
                    f'opacity: {opacity}; padding: 2px 4px; margin: 1px; '
                    f'border-radius: 3px; font-weight: bold;" '
                    f'title="{max_emotion}: {contribution:.3f}">{word}</span>'
                )
            else:
                html_parts.append(f'<span style="margin: 1px;">{word}</span>')
        else:
            html_parts.append(f'<span style="margin: 1px;">{word}</span>')
    
    return ' '.join(html_parts)

@st.cache_data
def analyze_with_shap(text, _model, _tokenizer, _emotion_names):
    """Analyze text with SHAP and return explanations"""
    try:
        # Use our custom word-level analysis instead of SHAP library
        word_contributions = get_word_level_shap_values(text, _model, _tokenizer, MAX_SEQUENCE_LENGTH)
        return word_contributions
        
    except Exception as e:
        st.error(f"Word-level analysis failed: {e}")
        return None

# Load resources
tokenizer, model = load_nlp_resources()

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
            
            # Display predicted emotion
            st.markdown(f"""
            <div class="result-container">
                <div class="emotion-result">
                    🎯 <strong>Predicted Emotion:</strong> <span style="color: #667eea; font-size: 1.3rem;">{max_emotion.upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create bar chart for emotion probabilities
            # Prepare data for visualization
            emotions = list(emotion_probs.keys())
            probabilities = list(emotion_probs.values())
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame({
                'Emotion': emotions,
                'Probability': probabilities
            }).sort_values('Probability', ascending=True)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color scheme - highlight the max emotion
            colors = ['#667eea' if emotion == max_emotion else '#a0a8d4' for emotion in df['Emotion']]
            
            bars = ax.barh(df['Emotion'], df['Probability'], color=colors)
            
            # Customize the plot
            ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Emotions', fontsize=12, fontweight='bold')
            ax.set_title('Emotion Analysis Results', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, max(probabilities) * 1.1)
            
            # Add value labels on bars
            for i, (bar, prob) in enumerate(zip(bars, df['Probability'])):
                ax.text(prob + max(probabilities) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.3f}', ha='left', va='center', fontweight='bold')
            
            # Style the plot
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
            # Set background color
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            
            # Display the chart
            st.pyplot(fig)
            
            # Clean up
            plt.close()
            
            # SHAP Analysis Section
            st.markdown("### 🔍 Word-Level Analysis")
            st.markdown("See which words contribute most to each emotion prediction:")
            
            with st.spinner("🧠 Analyzing word contributions..."):
                # Get word-level contributions for the input text
                word_contributions = analyze_with_shap(user_input, model, tokenizer, emotion_names_list)
                
                if word_contributions is not None and len(word_contributions) > 0:
                    # Create color-coded text
                    colored_html = create_colored_text_html(user_input, word_contributions, emotion_names_list, max_emotion)
                    
                    # Display word analysis
                    st.markdown(f"""
                    <div class="shap-container">
                        <h4 style="margin-top: 0; color: #333;">💡 Word Contributions</h4>
                        <div class="shap-text">
                            {colored_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add legend
                    st.markdown("""
                    <div class="legend-container">
                        <strong>Color Legend:</strong>
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
                    
                    # Display detailed word contributions
                    st.markdown("#### 📊 Detailed Word Contributions")
                    
                    # Create a DataFrame for word contributions
                    words = user_input.split()
                    word_data = []
                    
                    for i, word in enumerate(words):
                        if i < len(word_contributions):
                            for j, emotion in enumerate(emotion_names_list):
                                word_data.append({
                                    'Word': word,
                                    'Emotion': emotion,
                                    'Contribution': word_contributions[i, j]
                                })
                    
                    if word_data:
                        contrib_df = pd.DataFrame(word_data)
                        
                        # Show top contributing words for the predicted emotion
                        max_emotion_idx = emotion_names_list.index(max_emotion)
                        top_contrib = contrib_df[contrib_df['Emotion'] == max_emotion].nlargest(5, 'Contribution')
                        
                        st.markdown(f"**Top words contributing to '{max_emotion}' prediction:**")
                        for _, row in top_contrib.iterrows():
                            if abs(row['Contribution']) > 0.001:  # Lower threshold for visibility
                                st.write(f"• **{row['Word']}**: {row['Contribution']:.4f}")
                else:
                    st.warning("⚠️ Could not perform word-level analysis. Showing results without word explanations.")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.warning("Please ensure your model's input shape and prediction logic match your training setup.")
            
elif analyze_button and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")

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
