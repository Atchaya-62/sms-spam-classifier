import pickle
import string
import re
from flask import Flask, render_template, request
from nltk.corpus import stopwords
import nltk

# --- IMPORTANT: Ensure your folder structure is correct ---
# The 'templates' folder must be in the same directory as this 'app.py' file.
#
# your-project-folder/
# ├── templates/
# │   └── index.html
# ├── spam_classifier.pkl
# ├── tfidf_vectorizer.pkl
# └── app.py

# Create a Flask web application instance
app = Flask(__name__)

# --- Load the saved model and vectorizer ---
# We'll load the files that were saved in the Jupyter Notebook from Phase 1.
try:
    with open('spam_classifier.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    print("Error: Model or vectorizer files not found. Please ensure 'spam_classifier.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    # Exit the app or handle gracefully
    classifier = None
    tfidf_vectorizer = None

# --- Text Preprocessing Function ---
# This function must be exactly the same as the one used in the Jupyter Notebook.
def clean_text(text):
    """
    Cleans text by removing punctuation and stopwords.
    """
    # 1. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the home page with the text input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the user form."""
    if not classifier or not tfidf_vectorizer:
        return render_template('index.html', prediction="Error: Model not loaded. Check the terminal for more details.")
        
    # Get the message from the form
    message = request.form['message']

    # Clean the message using the same function from training
    cleaned_message = clean_text(message)

    # Transform the cleaned message into a TF-IDF vector
    # We must pass the text in a list format
    transformed_message = tfidf_vectorizer.transform([cleaned_message])

    # Make the prediction
    prediction = classifier.predict(transformed_message)
    
    # Get the prediction result (0 or 1)
    result = prediction[0]

    # Convert the numerical result back to 'Spam' or 'Not Spam'
    if result == 1:
        prediction_text = 'Spam'
    else:
        prediction_text = 'Not Spam'

    # Render the home page again, but this time with the prediction result
    return render_template('index.html', prediction=prediction_text)

# --- Run the App ---
if __name__ == '__main__':
    # This command starts the local web server.
    app.run(debug=True)
