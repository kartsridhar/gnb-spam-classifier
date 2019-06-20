import numpy as np
import pandas as pd         # to read the dataset as a df
import random
# To store the words in the message as a count, makes classification easier
from sklearn.feature_extraction.text import CountVectorizer
# To do the heavy lifting on Naive Bayes
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('data/training-data.csv', encoding="latin-1")

X = df['message'].values
y = df['class'].values

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

train_set, test_set, train_classes, test_classes = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(train_set, train_classes)
classifier.score(test_set, test_classes)

prediction = classifier.predict(test_set)
print(classification_report(test_classes, prediction))
