import numpy as np
import random

from sklearn import ensemble
from sklearn import metrics
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import graphviz
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string

stemmer = PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

pre_approved = []
with open('dictionary.txt', 'r') as f:
    newline = ""
    for line in f:
        stripped = line.strip()
        newline = stripped
        pre_approved.append(newline)


def create_unigram(x):
    lowerCase = x.lower()
    remove_punctuation = lowerCase.translate(remove_punctuation_map)
    tokenized = nltk.word_tokenize(remove_punctuation)
    filtered = [w for w in tokenized if w not in stopwords.words('english')]
    stemmed = []
    for item in filtered:
        stemmed.append(stemmer.stem(item))
    final_unigram = []
    for item in stemmed:
        for word in pre_approved:
            if item == word:
                final_unigram.append(item)
                break



    return final_unigram


reader = csv.reader(open('news-train.csv', 'r'))

next(reader)
documents = []

for row in reader:
    i, d, c = row
    documents.append({"id": i, "text": d, "category": c})


random.seed(0)
random.shuffle(documents)
num_data = int(len(documents) * 0.8)

train_data = documents[:num_data]
test_data = documents[num_data:]

train_x = []
train_y = []
test_x = []
test_y = []

for x in train_data:
    train_x.append(x.get("text"))
for x in train_data:
    train_y.append(x.get("category"))
for x in test_data:
    test_x.append(x.get("text"))
for x in test_data:
    test_y.append(x.get("category"))

print(len(train_data), len(test_data))

vectorizer = CountVectorizer()
train_x = vectorizer.fit_transform(train_x)
test_x = vectorizer.transform(test_x)

dtcf = tree.DecisionTreeClassifier()
dtcf = dtcf.fit(train_x, train_y)

pred = dtcf.predict(test_x)

print("Accuracy:", metrics.accuracy_score(test_y, pred))

forrest = ensemble.RandomForestClassifier()
forrest = forrest.fit(train_x, train_y)

predForrest = forrest.predict(test_x)

print("Accuracy Forrest:", metrics.accuracy_score(test_y, predForrest))

gradient = ensemble.GradientBoostingClassifier()
gradient = gradient.fit(train_x, train_y)

predGradient = gradient.predict(test_x)

print("Accuracy Gradient:", metrics.accuracy_score(test_y, predGradient))

dot_data = tree.export_graphviz(dtcf, max_depth=5, filled=True, rounded=True, feature_names=vectorizer.get_feature_names_out())
graph = graphviz.Source(dot_data)
graph.render("decision_tree")