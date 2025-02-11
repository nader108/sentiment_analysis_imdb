# IMDB Sentiment Analysis App

## 📌 Overview
This project is a **sentiment analysis application** for IMDB movie reviews, built using **Streamlit** and a pre-trained deep learning model. The app predicts whether a given review expresses a **positive** or **negative** sentiment.

## 🚀 Features
- **User-friendly Interface**: Enter a movie review and get an instant prediction.
- **Deep Learning Model**: Uses a pre-trained Keras model for sentiment analysis.
- **Text Preprocessing**: Cleans and normalizes input text before making predictions.
- **Real-time Predictions**: Displays sentiment results dynamically.

## 🛠️ Technologies Used
- **Python**
- **Streamlit**
- **TensorFlow/Keras**
- **NLTK** (Natural Language Processing)
- **Pickle** (to load the tokenizer)

## 📥 Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2️⃣ **Create a virtual environment and activate it**
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Download NLTK resources**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

5️⃣ **Run the Streamlit app**
```bash
streamlit run app.py
```

## 📌 Usage
- Enter a movie review in the text input box.
- Click on **Predict** to analyze sentiment.
- The model will return either **Positive 😊** or **Negative 😢** based on prediction.

## 📷 Demo
![Demo GIF](demo.gif)  

## 📂 Project Structure
```
├── imdb-sentiment-analysis/
│   ├── model_imdb.keras   # Pre-trained sentiment analysis model
│   ├── tokenizer.pkl      # Tokenizer used for text preprocessing
│   ├── app.py             # Streamlit app source code
│   ├── requirements.txt   # Dependencies
│   ├── README.md          # Project documentation
```

## 🙌 Contributing
Feel free to fork the repo and submit a pull request with improvements! 

## 📜 License
This project is licensed under the **MIT License**.

---
**⭐ Don't forget to star the repo if you find it useful!**
