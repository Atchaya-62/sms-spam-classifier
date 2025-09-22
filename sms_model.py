
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


print("First 5 rows of the dataset:")
print(df.head())
print("\nShape of the dataset:", df.shape)

import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
import re

def clean_text(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply the cleaning function to the 'message' column
df['cleaned_message'] = df['message'].apply(clean_text)

# Display a sample of the cleaned data
print("\nOriginal vs. Cleaned message:")
print("Original:", df['message'][0])
print("Cleaned:", df['cleaned_message'][0])

# <codecell>
# Step 3.3: Transform the Data
# We will use TF-IDF to convert our cleaned text into a numerical format.

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer instance
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned messages.
# This learns the vocabulary and IDF, then transforms the text into a matrix.
X = tfidf_vectorizer.fit_transform(df['cleaned_message'])
y = df['label'].map({'ham': 0, 'spam': 1}) # Convert labels to numerical format (0 for ham, 1 for spam)

print("\nShape of the TF-IDF matrix:", X.shape)
print("A sample of the TF-IDF matrix (for the first message):")
print(X[0])

# <codecell>
# Step 3.4: Train the Model
# Now, we split our data and train a Multinomial Naive Bayes classifier.

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# <codecell>
# Step 3.5: Test and Save
# We'll evaluate the model's performance and then save it for the Flask app.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


with open('spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully as 'spam_classifier.pkl' and 'tfidf_vectorizer.pkl'.")

