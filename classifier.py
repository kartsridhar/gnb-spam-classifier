import numpy as np
import pandas as pd         # to read the dataset as a df
import random
# To store the words in the message as a count, makes classification easier
from sklearn.feature_extraction.text import CountVectorizer
# To do the heavy lifting on Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# To make splitting of train/test data easier
from sklearn.model_selection import train_test_split

# Function to calculate the accuracy
def calc_accuracy(gt_labels, pred_labels):
    same = 0
    for i in range(len(gt_labels)):
        if gt_labels[i] == pred_labels[i]:
            same += 1
    accuracy = (same/len(gt_labels)) * 100
    return accuracy

df = pd.read_csv('data/training-data.csv', encoding="latin-1")

messages = df['message'].values
classes = df['class'].values

vectorizer = CountVectorizer()
messages = vectorizer.fit_transform(messages)

# random.shuffle(messages)
train_set, test_set, train_classes, test_classes = train_test_split(messages, classes, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(train_set, train_classes)
classifier.score(test_set, test_classes)

predictions = classifier.predict(test_set)
accuracy = calc_accuracy(test_classes, predictions)
print(accuracy)
