# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:47:27 2017

@author: mzent
"""
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import csv
import pickle

train = []
test = []
with open('trainingData5.csv', newline='', encoding="latin-1") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train.append((row[0] + ": " + row[2],row[3]))

with open('testData5.csv', newline='', encoding="latin-1") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test.append((row[0] + ": " + row[2],row[3]))

print("read data")
cl = NaiveBayesClassifier(train)
print("created classifier")
# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

# Show 5 most informative features
cl.show_informative_features(25)

print(pickle.dump(cl))