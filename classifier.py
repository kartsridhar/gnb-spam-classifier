import numpy as np
import pandas as pd
# To store the words in the message as a count, makes classification easier
from sklearn.feature_extraction.text import CountVectorizer
# To do the heavy lifting on Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# To make splitting of train/test data easier
from sklearn.model_selection import train_test_split
from Flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# Function to calculate the accuracy
# def calc_accuracy(gt_labels, pred_labels):
#     same = 0
#     for i in range(len(gt_labels)):
#         if gt_labels[i] == pred_labels[i]:
#             same += 1
#     accuracy = (same/len(gt_labels)) * 100
#     return accuracy

@app.route('/classify', methods=['POST'])
def classify():
    df = pd.read_csv('data/training-data.csv', encoding="latin-1")

    messages = df['message'].values
    classes = df['class'].values

    vectorizer = CountVectorizer()
    messages = vectorizer.fit_transform(messages)

    train_set, test_set, train_classes, test_classes = train_test_split(messages, classes, test_size=0.3, random_state=42, shuffle=True)

    classifier = MultinomialNB()
    classifier.fit(train_set, train_classes)
    classifier.score(test_set, test_classes)
    # predictions = classifier.predict(test_set)
    # accuracy = calc_accuracy(test_classes, predictions) ~ 97%

    # Section to get the message from HTML
    if (request.method == 'POST'):
        message = request.form['message']
        text = [message]
        text_transform = vectorizer.transform(text)
        prediction = classifier.predict(text_transform.toarray())
    return render_template('result.html', prediction=prediction)
