import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import csv

# Importing and preparing datasets

print('Importing training data')
directory = "/home/iad/Documents/ENSTA/3A/rob_311/TP_rob311/TP4/" # data directory on iad's computer 
# directory = "./" # data directory on madeleine's computer 
# directory = "yourpath/" # prof : put your data directory here
datafileTrain = "mnist_train.csv"
dataFileTest = "mnist_test.csv"

# choose if you want to have a nice plot or just to print the result in the terminal
plotPrettyConfMatrix = True


train_data = np.empty((60000,784),dtype=int)
train_class = np.empty(60000,dtype=int)


file = open(directory+datafileTrain, "r")
csvFile = csv.reader(file)
i = -1
for row in csvFile:
    if i > -1:
        train_class[i] = np.array(row[0])
        train_data[i][:] = np.array([row[1:]])
    i += 1


print(train_data.shape)

print('Importing testing data')

test_data = np.empty((10000, 784), dtype=int)
test_class = np.empty(10000, dtype=int)

file = open(directory+dataFileTest, "r")
csvFile = csv.reader(file)
i = -1
for row in csvFile:
    if i > -1:
        test_class[i] = np.array(row[0])
        test_data[i][:] = np.array([row[1:]])
    i += 1

print(test_data.shape)
print(test_class[:10])

# Reducting the number of components using PCA

print('Reducing dimensions of data')

pca = PCA(n_components=50)
pca.fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)
print(train_data.shape)

# Creating and training SVM Classifier

print('Training classifier')

clf = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))
clf.fit(train_data, train_class)

# Use the trained classifier to classify the test dataset

print('Using trained classifier')

test_predict = clf.predict(test_data)
print('Classes from test dataset : ')
print(test_class[:10])
print('Predicted classes :')
print(test_predict[:10])


# compute the confusion matrix and the accuracy
conf_mat = confusion_matrix(test_class, test_predict)
nClass = conf_mat.shape[0]
accuracy = 0
for ii in range(nClass):
    accuracy += conf_mat[ii][ii]
accuracy = accuracy/test_class.size


# display results
print("accuracy :")
print(accuracy)
 
if not plotPrettyConfMatrix:
    print("here is the confusion matrix :")
    print(conf_mat)
else:
    print("using plot_confusion_matrix to have a nice plot")
    np.set_printoptions(precision=2)
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_data, test_class,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    
    plt.show()
    print("done")
