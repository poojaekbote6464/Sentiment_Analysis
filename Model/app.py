from flask import Flask, render_template, request
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained model
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess user input review text
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text)  
        text = re.sub(r'\d+', '', text)  
        text = text.lower() 
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        normalized_text = [lemmatizer.lemmatize(word) for word in filtered_text]

        return ' '.join(normalized_text)
    else:
        return ''

# Function to predict sentiment
def predict_sentiment(review_text):
    try:
        preprocessed_text = preprocess_text(review_text)
        sentiment = model.predict([preprocessed_text])[0]
        return sentiment
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form['review_text']
        sentiment = predict_sentiment(review_text)
        return render_template('results.html', sentiment=sentiment)
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('error.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4552)
