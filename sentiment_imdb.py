import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import html
import unicodedata
import string
import pickle as pkl


model = load_model("model_imdb.keras")

# Load the tokenizer used during training
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pkl.load(handle)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Text preprocessing functions
def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words):
    return [word for word in words if word not in stop_words]


def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]


def text2words(text):
    return word_tokenize(text)


def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words)
    words = lemmatize_words(words)
    return ' '.join(words)


# Streamlit app
st.title("IMDB Sentiment Analysis")

# User input
test_review = st.text_input("Enter a review:")

if st.button("Predict"):
    if test_review:
        # Preprocess the input text
        processed_text = normalize_text(test_review)

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([processed_text])

        # Pad the sequence
        max_length = 100  # Ensure this matches the max_length used during training
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Make prediction
        prediction = model.predict(padded_sequence)
        sentiment = "positive 😊" if prediction > 0.5 else "negative 😢"

        # Display result
        st.write(f"Predicted sentiment: {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")