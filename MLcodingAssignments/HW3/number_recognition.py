#!/usr/bin/env python

import sys

def decision_tree(train, test):
    y = []
    #Your code here
    return y

def knn(train, test):
    y = []
    #Your code here
    return y

def neural_net(train, test):
    y = []
    #Your code here
    return y

def svm(train, test):
    y = []
    #Your code here
    return y

def pca_knn(train, test):
    y = []
    #Your code here
    return y

def pca_svm(train, test):
    y = []
    #Your code here
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]

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
