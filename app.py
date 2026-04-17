import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# 1. Page Config (Changed to Centered for better organization)
# ------------------------------
st.set_page_config(page_title="AI Predictor", page_icon="⚡", layout="centered")

# ------------------------------
# 2. Load Resources
# ------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔍 DEBUG (ADD THIS)
st.write("BASE_DIR:", BASE_DIR)
st.write("FILES:", os.listdir(BASE_DIR))

@st.cache_resource
def load_resources():
    model = load_model(
        os.path.join(BASE_DIR, "lstm_model.h5"),
        compile=False,
        safe_mode=False
    )

    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)

    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

    return model, tokenizer, max_len, reverse_word_index


# ❗ ADD PROPER ERROR HANDLING
try:
    model, tokenizer, max_len, reverse_word_index = load_resources()
except Exception as e:
    st.error(f"Actual error: {e}")
    st.stop()

# ------------------------------
# 3. Prediction & Callback
# ------------------------------
def predict_next_words(text, top_n=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    if not sequence:
        return []
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
    preds = model.predict(sequence, verbose=0)[0]
    top_indices = preds.argsort()[-top_n:][::-1]
    return [reverse_word_index.get(i, "") for i in top_indices if i != 0]

if "text_box" not in st.session_state:
    st.session_state.text_box = ""

def add_suggestion(word):
    current_text = st.session_state.text_box
    if current_text and not current_text.endswith(" "):
        st.session_state.text_box += " " + word + " "
    else:
        st.session_state.text_box += word + " "

# ------------------------------
# 4. ToolsHub Visual Identity CSS
# ------------------------------
st.markdown("""
<style>
/* Import heavy geometric font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;900&display=swap');

/* Force deep dark mode on the whole app */
.stApp {
    background-color: #080808 !important;
    background-image: radial-gradient(circle at 50% 0%, #1a0f05 0%, #080808 60%) !important;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* The Massive Header (Solid + Outline) */
.hero-container {
    margin-top: 50px;
    margin-bottom: 20px;
    text-align: center;
}
.hero-solid {
    font-size: 80px;
    font-weight: 900;
    line-height: 0.85;
    color: #ffffff;
    letter-spacing: -2px;
}
.hero-outline {
    font-size: 80px;
    font-weight: 900;
    line-height: 0.85;
    color: transparent;
    -webkit-text-stroke: 2px #ff6b00; /* ToolsHub Orange */
    letter-spacing: -2px;
    display: block;
}

.sub-headline {
    font-size: 18px;
    color: #a3a3a3;
    font-weight: 400;
    margin-top: 25px;
    margin-bottom: 50px;
    line-height: 1.4;
    text-align: center;
}

/* Input Field Styling */
div[data-baseweb="input"] > div {
    background-color: transparent;
    border: none;
    border-bottom: 2px solid #333333;
    border-radius: 0px;
    transition: all 0.3s ease;
}
div[data-baseweb="input"] > div:focus-within {
    border-bottom: 2px solid #ff6b00;
    box-shadow: none;
}
input {
    color: #ffffff !important;
    font-size: 22px !important;
    padding: 15px 0px !important;
    font-weight: 600;
    text-align: center !important;
}
input::placeholder { color: #444444 !important; text-align: center !important; }

/* Suggestion Buttons - Perfectly Organized Chips */
div.stButton > button {
    width: 100%;
    background-color: #121212;
    color: #ffffff;
    border: 1px solid #333333;
    font-weight: 600;
    font-size: 16px;
    padding: 12px;
    border-radius: 8px;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background-color: #ff6b00;
    color: #000000;
    border-color: #ff6b00;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 5. Layout Engine
# ------------------------------

# Massive Typography Header
st.markdown("""
    <div class="hero-container">
        <span class="hero-solid">NEXT</span><br>
        <span class="hero-outline">WORDS.</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="sub-headline">
        Simple, powerful prediction. No trackers, no setup - just results.
    </div>
""", unsafe_allow_html=True)

# Input Field
st.text_input(
    label="Type your sentence",
    key="text_box",
    placeholder="START TYPING...",
    label_visibility="collapsed"
)

user_input = st.session_state.text_box

# Live Suggestions
if user_input.strip():
    suggestions = predict_next_words(user_input, 3)
    st.write("") # Spacer
    
    if suggestions:
        # Filters out any empty string suggestions just in case
        valid_suggestions = [word for word in suggestions if word]
        
        # Creates perfectly sized, evenly spaced columns for the exact number of words
        btn_cols = st.columns(len(valid_suggestions))
        
        for i, word in enumerate(valid_suggestions):
            btn_cols[i].button(
                word, 
                key=f"suggest_{i}", 
                on_click=add_suggestion, 
                args=(word,),
                use_container_width=True
            )
