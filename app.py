from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load your Keras model
model = load_model('best_model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Flask app initialization
app = Flask(__name__)

# Define the predict_class function
def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    
    # Transform text to a sequence of integers using the tokenizer object
    xt = tokenizer.texts_to_sequences([text])
    print(f"Tokenized text: {xt}")  # Debug: print the tokenized text
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    print(f"Padded sequences: {xt}")  # Debug: print the padded sequences
    # Do the prediction using the loaded model
    yt = model.predict(xt)
    print(f"Model prediction: {yt}")  # Debug: print the raw model prediction
    yt_class = yt.argmax(axis=1)
    print(f"Predicted class: {yt_class}")  # Debug: print the predicted class
    # Get the predicted sentiment
    sentiment = sentiment_classes[yt_class[0]]
    # Get the confidence of the prediction
    confidence = np.max(yt)
    
    return sentiment, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if not request.form or 'text' not in request.form:
        return render_template('index.html', error='Invalid input')
    
    text = request.form['text']
    
    # Perform sentiment analysis using the predict_class function
    sentiment, confidence = predict_class(text)
    
    return render_template('result.html', text=text, sentiment=sentiment, confidence=confidence*100)

if __name__ == '__main__':
    app.run(debug=True)
