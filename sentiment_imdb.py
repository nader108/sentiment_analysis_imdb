import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')

# Load model
try:
    model = load_model("model_imdb.keras")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    model = None

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
                return ' '.join(word_tokenize(text.lower()))  # Basic preprocessing for demonstration

            processed_text = preprocess_text(test_review)

            # Convert text to sequences
            sequence = tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

            # Predict sentiment
            prediction = model.predict(padded_sequence)
            sentiment = "positive 😊" if prediction > 0.5 else "negative 😢"
            st.write(f"Predicted sentiment: {sentiment}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
