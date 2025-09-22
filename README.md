SMS Spam Classifier

Overview
A beginner-friendly machine learning project that uses a Naive Bayes model to classify SMS messages as "spam" or "not spam." This project is built as an end-to-end web application using Flask.

Technologies Used
Python: The core programming language.

Pandas: For data loading and manipulation.

NLTK & Scikit-learn: For text preprocessing and the machine learning model.

Flask: A lightweight web framework for the web application.

Project Structure
sms_spam_classifier.ipynb: The Jupyter Notebook containing all the data cleaning and model training steps.

app.py: The Flask web application backend.

templates/index.html: The user interface for the web app.

spam_classifier.pkl: The saved machine learning model.

tfidf_vectorizer.pkl: The saved TF-IDF vectorizer.

How to Run the Project
Clone this repository: git clone https://github.com/your-username/sms-spam-classifier.git

Install the required libraries: pip install -r requirements.txt (You'll need to create this file by running pip freeze > requirements.txt on your local machine)

Run the Flask application: python app.py

My Learning Journey
This project was a great introduction to the full machine learning pipeline, from data preparation to model deployment. I learned key concepts like text preprocessing, TF-IDF vectorization, and model serialization.