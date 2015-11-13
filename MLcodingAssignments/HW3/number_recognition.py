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


# def read_in_data(trainpath, testpath):
#     datapath = "image_data"
#     train_datapath = os.path.join(datapath, trainpath)
#     trainfile = open(train_datapath, 'r')

#     test_datapath = os.path.join(datapath, testpath)

#     testfile = open(test_datapath, 'r')

#     xtrain = []
#     ytrain = []
#     xtrainline = []

#     print "processing training data..."
  
#     while(1):

#         # if len(xtrainline) and len(xtrain):
#         #     xtrain = np.vstack((xtrain, xtrainline))
#         line = trainfile.readline()

#         if not line:
#             break

#         line = line.strip().split(' ')
#         ytrain.append(line[0])
#         xtrainline = np.asarray(line[1:])
#         xtrainline = xtrainline.astype(np.float)
#         if not len(xtrain):
#             xtrain = xtrainline
#         else:
#             xtrain = np.vstack((xtrain, xtrainline))
#         # print xtrain
#     ytrain = np.asarray(ytrain).astype(np.float)

#     xtest = []
#     ytest = []
#     xtestline = []


#     print "processing testing data..."
#     while(1):
#         # if len(xtestline) and len(xtest):
#         #     xtest = np.vstack((xtest, xtestline))
#         line = testfile.readline()

#         if not line:
#             break

#         line = line.strip().split(' ')
#         ytest.append(line[0])
#         xtestline = np.asarray(line[1:])
#         xtestline = xtestline.astype(np.float)
#         if not len(xtest):
#             xtest = xtestline
#         else:
#             xtest = np.vstack((xtest, xtestline))

#     ytest = np.asarray(ytest).astype(np.float)

#     trainfile.close()
#     testfile.close()
#     return xtrain, ytrain, xtest, ytest


def read_in_data(trainpath, testpath):   # V2, modify as read in the data once and then process it
    # datapath = "image_data"
    # train_datapath = os.path.join(datapath, trainpath)
    # trainfile = open(train_datapath, 'r')
    trainfile = open(trainpath, 'r')

    # test_datapath = os.path.join(datapath, testpath)

    # testfile = open(test_datapath, 'r')
    testfile = open(testpath, 'r')

    xtrain = []
    ytrain = []

    rawparse = []


    print "processing train data..."

    while(1):
        line = trainfile.readline()
        if not line:
            break

        line = line.strip().split(' ')
        rawparse.append(line)

    rawparse = np.asarray(rawparse).astype(np.float)
    # print type(rawparse)
    # print len(rawparse)
    # print len(rawparse[0])
    # print rawparse[0,0]
    # print rawparse[1,0]

    ytrain = rawparse[ : , 0]
    xtrain = rawparse[: , 1:]

    print "processing test data..."

    rawparse = []
    while(1):
        line = testfile.readline()
        if not line:
            break

        line = line.strip().split(' ')
        rawparse.append(line)

    rawparse = np.asarray(rawparse).astype(np.float)

    ytest = rawparse[ : , 0]
    xtest = rawparse[: , 1:]


    trainfile.close()
    testfile.close()
    return xtrain, ytrain, xtest, ytest
    # return

def decision_tree(train, test):



    y = []
    #Your code here
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    xtrain = train[0]
    ytrain = train[1]
    xtest = test[0]
    ytest = test[1]


    # for criter in ('gini', 'entropy'):
    #     for mDep in (3, 4, 5, None):
    #         for maxfeat in (None, 'sqrt', 'log2'):
    #             Tclassifier = DecisionTreeClassifier(criterion = 'gini', max_depth = mDep, max_features = maxfeat)
    #             Tclassifier.fit(xtrain,ytrain)
    #             y.append(Tclassifier.score(xtest, ytest))
    Tclassifier = DecisionTreeClassifier()
    Tclassifier.fit(xtrain,ytrain)
    y.append(Tclassifier.score(xtest, ytest))
   
    return y

def knn(train, test):
    y = []
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    terror = []



    xtrain = train[0]
    ytrain = train[1]
    xtest = test[0]
    ytest = test[1]

    # for neighbors in (1,10,50,100,200):
    #     print neighbors
    # # (range(1, 11), 50, 100, 200):
    #     for weight in ('uniform', 'distance'):
    #         knnClassifier = KNeighborsClassifier(n_neighbors = neighbors, weights = weight)
    #         knnClassifier.fit(xtrain,ytrain)
    #         y.append(knnClassifier.score(xtest, ytest))
    #         print "test error done"
    #         terror.append(knnClassifier.score(xtrain, ytrain))
    #         print "train error done"
    # print y
    # print terror

    #Your code here

    knnClassifier = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
    knnClassifier.fit(xtrain,ytrain)
    y.append(knnClassifier.score(xtest, ytest))
    return y

def neural_net(train, test):
    y = []
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    xtrain = train[0]
    ytrain = train[1]
    xtest = test[0]
    ytest = test[1]

    n_netClassifier = Perceptron()
    n_netClassifier.fit(xtrain,ytrain)
    y = n_netClassifier.score(xtest, ytest)
    #Your code here
    return y

def svm(train, test):
    y = []
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    xtrain = train[0]
    ytrain = train[1]
    xtest = test[0]
    ytest = test[1]

    # for ker in ('rbf', 'linear', 'poly'):
    #     svmClassifier = SVC(kernel = ker)
    #     svmClassifier.fit(xtrain,ytrain)
    #     y.append(svmClassifier.score(xtest, ytest))
    svmClassifier = SVC(kernel = 'poly')
    svmClassifier.fit(xtrain,ytrain)
    y.append(svmClassifier.score(xtest, ytest))


    #Your code here
    return y

def pca_knn(train, test):
    y = []
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)
    xtrain = train[0]
    ytrain = train[1]
    xtest = test[0]
    ytest = test[1]

    # for dim in (5, 10, 100, 200):
    # for scale in range(1, 30):
    #     dim = 1 * scale

    #     pca = RandomizedPCA(n_components = dim)

    #     xtrainReduced = pca.fit_transform(xtrain)
    #     xtestReduced = pca.fit_transform(xtest)

        
    #     knnClassifier = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
    #     knnClassifier.fit(xtrainReduced,ytrain)
    #     y.append(knnClassifier.score(xtestReduced, ytest))

    #Your code here

    pca = RandomizedPCA(n_components = 5)

    xtrainReduced = pca.fit_transform(xtrain)
    xtestReduced = pca.fit_transform(xtest)

    
    knnClassifier = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
    knnClassifier.fit(xtrainReduced,ytrain)
    y.append(knnClassifier.score(xtestReduced, ytest))

    return y

def pca_svm(train, test):
    y = []
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)

    xtrain = train[0]
    ytrain = train[1]
    xtest = test[0]
    ytest = test[1]
    # for dim in (5, 10, 200):
    #     pca = RandomizedPCA(n_components = 5)
    #     xtrainReduced = pca.fit_transform(xtrain)
    #     xtestReduced = pca.fit_transform(xtest)

    #     svmClassifier = SVC(kernel = 'poly')
    #     svmClassifier.fit(xtrainReduced,ytrain)
    #     y.append(svmClassifier.score(xtestReduced, ytest))
    #Your code here

    pca = RandomizedPCA(n_components = 5)
    xtrainReduced = pca.fit_transform(xtrain)
    xtestReduced = pca.fit_transform(xtest)

    svmClassifier = SVC(kernel = 'poly')
    svmClassifier.fit(xtrainReduced,ytrain)
    y.append(svmClassifier.score(xtestReduced, ytest))
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]


    # hard code for testing and debug

    # model = "pcasvm"

    # train = "zip.train"
    # test = "zip.test"

    # read_in_data(train , test)
    xtrain, ytrain, xtest, ytest = read_in_data(train, test)

    train = (xtrain, ytrain)
    test = (xtest, ytest)

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

    #-------------------------------------


    # test and experiment

    # combo_result = []
    # # combo_result.append("decision_tree")
    # # combo_result.append(decision_tree(train, test))
    # # combo_result.append("knn")
    # # combo_result.append(knn(train, test))
    # # combo_result.append("neural_net")
    # # combo_result.append(neural_net(train, test))
    # # combo_result.append("svm")
    # # combo_result.append(svm(train, test))
    # combo_result.append("pca_knn")
    # combo_result.append(pca_knn(train, test))
    # # combo_result.append("pca_svm")
    # # combo_result.append(pca_svm(train, test))

    # # outfile = open("result_combo.txt", 'w')
    # # for item in combo_result:
    # #     outfile.write(item)
    # #     outfile.write('\n')
    # # outfile.close()

    # print combo_result






