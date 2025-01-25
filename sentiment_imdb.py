import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import html
import unicodedata
import string
import pickle as pkl

# Ensure NLTK data is downloaded
nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
try:
    model = tf.keras.models.load_model("model_imdb.keras")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load tokenizer
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pkl.load(handle)
except FileNotFoundError:
    st.error("Tokenizer file not found. Please check its path.")
    tokenizer = None

# Streamlit app
st.title("IMDB Sentiment Analysis")

test_review = st.text_input("Enter a review:")

if st.button("Predict"):
    if not model or not tokenizer:
        st.error("Model or tokenizer is not loaded properly.")
    elif test_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        try:
            # Preprocessing input
            def preprocess_text(text):
                st.write(f"Original Text: {text}")
                # Tokenization and lowercasing
                tokens = word_tokenize(text.lower())
                st.write(f"Tokenized Text: {tokens}")
                return ' '.join(tokens)

            processed_text = preprocess_text(test_review)

            # Convert text to sequences
            sequence = tokenizer.texts_to_sequences([processed_text])
            st.write(f"Sequence: {sequence}")

            # Pad the sequence
            padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
            st.write(f"Padded Sequence: {padded_sequence}")

            # Predict sentiment
            prediction = model.predict(padded_sequence)
            st.write(f"Raw Prediction: {prediction}")

            sentiment = "positive 😊" if prediction > 0.5 else "negative 😢"
            st.write(f"Predicted sentiment: {sentiment}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")