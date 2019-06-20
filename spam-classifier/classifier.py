import numpy as np
import pandas as pd         # to read the dataset as a df
import random
# To store the words in the message as a count, makes classification easier
from sklearn.feature_extraction.text import CountVectorizer
# To do the heavy lifting on Naive Bayes
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/training-data.csv', encoding="latin-1")

vectorizer = CountVectorizer()
df['message'] = vectorizer.fit_transform(df['message'])

train_set, test_set, train_classes, test_classes = train_test_split(df['message'], df['class'], test_size=0.3, random_state=42)
