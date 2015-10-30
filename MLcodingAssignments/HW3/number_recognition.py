#!/usr/bin/env python


#Q2 for the ML HW3
# jinlong Feng-jf5rb


import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA



# >>> clf = DecisionTreeClassifier(random_state=0)
# >>> iris = load_iris()
# >>> cross_val_score(clf, iris.data, iris.target, cv=10)


def read_in_data(trainpath, testpath):
    datapath = "image_data"
    train_datapath = os.path.join(datapath, trainpath)
    trainfile = open(train_datapath, 'r')

    test_datapath = os.path.join(datapath, testpath)

    testfile = open(test_datapath, 'r')

    xtrain = []
    ytrain = []
    xtrainline = []

    print "processing training data..."
  
    while(1):

        # if len(xtrainline) and len(xtrain):
        #     xtrain = np.vstack((xtrain, xtrainline))
        line = trainfile.readline()

        if not line:
            break

        line = line.strip().split(' ')
        ytrain.append(line[0])
        xtrainline = np.asarray(line[1:])
        xtrainline = xtrainline.astype(np.float)
        if not len(xtrain):
            xtrain = xtrainline
        else:
            xtrain = np.vstack((xtrain, xtrainline))
        # print xtrain
    ytrain = np.asarray(ytrain).astype(np.float)

    xtest = []
    ytest = []
    xtestline = []


    print "processing testing data..."
    while(1):
        # if len(xtestline) and len(xtest):
        #     xtest = np.vstack((xtest, xtestline))
        line = testfile.readline()

        if not line:
            break

        line = line.strip().split(' ')
        ytest.append(line[0])
        xtestline = np.asarray(line[1:])
        xtestline = xtestline.astype(np.float)
        if not len(xtest):
            xtest = xtestline
        else:
            xtest = np.vstack((xtest, xtestline))

    ytest = np.asarray(ytest).astype(np.float)

    trainfile.close()
    testfile.close()
    return xtrain, ytrain, xtest, ytest


def decision_tree(train, test):



    y = []
    #Your code here
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)


    Tclassifier = DecisionTreeClassifier()
    Tclassifier.fit(xtrain,ytrain)
    y = Tclassifier.score(xtest, ytest)

   
    return y

def knn(train, test):
    y = []
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    
    for i in (1,3,5,7,9,11):
        knnClassifier = KNeighborsClassifier(n_neighbors = i)
        knnClassifier.fit(xtrain,ytrain)
        y.append(knnClassifier.score(xtest, ytest))

    #Your code here
    return y

def neural_net(train, test):
    y = []
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)

    n_netClassifier = Perceptron()
    n_netClassifier.fit(xtrain,ytrain)
    y = n_netClassifier.score(xtest, ytest)
    #Your code here
    return y

def svm(train, test):
    y = []
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)

    svmClassifier = SVC()
    svmClassifier.fit(xtrain,ytrain)
    y = svmClassifier.score(xtest, ytest)


    #Your code here
    return y

def pca_knn(train, test):
    y = []
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    print xtrain
    print len(xtrain)
    print len(xtrain[0])
    pca = RandomizedPCA(n_components = 5)
    xtrain = pca.transform(xtrain)
    print xtrain
    print len(xtrain)
    print len(xtrain[0])
    # y = pca_knnClassifier.score(xtest, ytest)
    #Your code here
    return y

def pca_svm(train, test):
    y = []
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    print xtrain
    print xtrain.shape()
    pca_knnClassifier = RandomizedPCA(n_components = 5)
    xtrain = pca_knnClassifier.fit_transform(xtrain)
    #Your code here
    return y

if __name__ == '__main__':
    # model = sys.argv[1]
    # train = sys.argv[2]
    # test = sys.argv[3]


    # hard code for testing and debug

    model = "pcaknn"
    train = "zip.train"
    test = "zip.test"


    #----------------------------



    if model == "dtree":
        print(decision_tree(train, test))
    elif model == "knn":
        print(knn(train, test))
    elif model == "net":
        print(neural_net(train, test))
    elif model == "svm":
        print(svm(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    elif model == "pcasvm":
        print(pca_svm(train, test))
    else:
        print("Invalid method selected!")
