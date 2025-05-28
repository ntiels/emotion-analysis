import streamlit as st
import pickle
import re
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

st.set_page_config(page_title="NLP Emotion Analyzer", layout="centered")

MAX_SEQUENCE_LENGTH = 100
emotion_categories = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'anger', 6:'love'}
emotion_names_list = [emotion_categories[i] for i in range(len(emotion_categories))]

# cached for efficiency
@st.cache_resource
def load_nlp_resources():
    """
    Loads the tokenizer and model, caching them for efficiency.
    """
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    model = load_model('model.keras')
    return tokenizer, model

tokenizer, model = load_nlp_resources()

def preprocess_text_app(text, tokenizer, max_sequence_length=100):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"[^a-z0-9\s!?]", "", cleaned_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# shap pipeline
pred = transformers.pipeline(
    "emotion-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,
    return_all_scores=True,
)

explainer = shap.Explainer(pred)
shap_values = explainer(user_input)
shap.plots.text(shap_values)

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
