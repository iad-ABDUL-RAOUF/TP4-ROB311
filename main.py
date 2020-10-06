import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import csv

#Importing and preparing datasets

print('Importing training data')

train_data = np.empty((0,784),dtype=int)
train_class = np.empty((0),dtype=int)

file=open("./mnist_train.csv","r")
test=csv.reader(file)
i=0;
for row in test:
    if i >0:
        train_class = np.append(train_class, np.array(row[0]))
        train_data = np.append(train_data, np.array([row[1:]]), axis=0)
    i+=1

print(train_data.shape)

print('Importing testing data')

test_data = np.empty((0,784),dtype=int)
test_class = np.empty((0),dtype=int)

file=open("./mnist_test.csv","r")
test=csv.reader(file)
firstline=True
for row in test:
    if firstline: #The first line is the names of the columns, we don't store it.
        firstline = False
    else:
        test_class = np.append(test_class, np.array(row[0]))
        test_data = np.append(test_data, np.array([row[1:]]), axis=0)

print(test_data.shape)

# Reducting the number of components using PCA

print('Reducing dimensions of data')

pca = PCA(n_components=50)
pca.fit(train_data)
pca.transform(train_data)
pca.transform(test_data)

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