import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import csv

#Importing and preparing datasets

print('Importing training data')

train_data = []
train_class = []

file=open("./mnist_train.csv","r")
test=csv.reader(file)
firstline=True
for row in test:
    if firstline: #The first line is the names of the columns, we don't store it.
        firstline = False
    else:
        train_class.append(row[0])
        train_data.append(row[1:])

print('Importing testing data')

test_data = []
test_class = []

file=open("./mnist_test.csv","r")
test=csv.reader(file)
firstline=True
for row in test:
    if firstline: #The first line is the names of the columns, we don't store it.
        firstline = False
    else:
        test_class.append(row[0])
        test_data.append(row[1:])

#Creating and training SVM Classifier

print('Training classifier')

clf = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))
clf.fit(train_data,train_class)

# Use the trained classifier to classify the test dataset

print('Using trained classifier')

test_predict = clf.predict(test_class)
print('Classes from test dataset : ')
print(test_class[:10])
print('Predicted classes :')
print(test_predict[:10])