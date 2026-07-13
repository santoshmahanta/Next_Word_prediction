import streamlit as st

# MUST be the very first Streamlit command in the script
st.set_page_config(page_title="Next Word Prediction", layout="centered")

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model (1).h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Single next-word prediction
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    if predicted_index == 0:
        return ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# ------------------------------
# Full sentence generation (loops the next-word prediction)
# ------------------------------
def generate_sentence(seed_text, num_words=10):
    result_text = seed_text
    for _ in range(num_words):
        next_word = predict_next_word(result_text)
        if next_word == "":
            # model has nothing more to add (predicted padding/unknown) - stop early
            break
        result_text += " " + next_word
    return result_text

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Next Word / Sentence Prediction (LSTM)")
st.write("Enter a starting phrase and the model will keep predicting words to build a full sentence.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a starting phrase here...")

num_words = st.slider("Number of words to generate", min_value=1, max_value=50, value=10)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        generated = generate_sentence(user_input, num_words=num_words)
        st.success(f"**Generated Text:** {generated}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word / Sentence Prediction using Streamlit")