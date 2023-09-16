from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load the model
try:
    loaded_model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    loaded_model = None

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
tfidf_v = TfidfVectorizer()
app = Flask(__name__, template_folder='./templates', static_folder='./static')

def fake_news_det(news):
    corpus = []  # Create a local list for each prediction
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    for word in review:
        if word not in stpwrds:
            corpus.append(lemmatizer.lemmatize(word))
    input_data = [' '.join(corpus)]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return "Looking Fake⚠ News📰" if prediction[0] == 0 else "Looking Real News📰"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        if loaded_model:
            pred = fake_news_det(message)
            return render_template('index.html', prediction=pred)
        else:
            return render_template('index.html', prediction="Model not found")
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
