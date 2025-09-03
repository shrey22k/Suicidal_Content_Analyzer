from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Initialize Flask App ---
app = Flask(__name__)
# Enable CORS to allow communication from the frontend
CORS(app) 

# --- Load Pre-trained Model and Vectorizer ---
try:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("\nERROR: 'model.joblib' or 'vectorizer.joblib' not found.")
    print("Please run the 'train.py' script first to train and save the model.")
    exit()

# --- Text Preprocessing Function (Must be IDENTICAL to the one in train.py) ---
def preprocess_text(text):
    """Cleans and preprocesses a single text string."""
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    return " ".join(words)

# --- Define API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives text input and returns a prediction from the model."""
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing text in request'}), 400

    # Get text from the POST request
    user_text = request.json['text']
    
    if not user_text.strip():
        return jsonify({'prediction': 'No input', 'class': 'neutral'})

    # Preprocess the text
    processed_text = preprocess_text(user_text)

    # Vectorize the preprocessed text
    vectorized_text = vectorizer.transform([processed_text])

    # Make a prediction
    prediction = model.predict(vectorized_text)[0] # Get the first (and only) prediction

    # Return the prediction as JSON
    response = {'prediction': f'Model classified as: {prediction}', 'class': prediction}
    return jsonify(response)

# --- Run the Flask App ---
if __name__ == '__main__':
    print("Starting Flask server... Go to your web app to send requests.")
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)