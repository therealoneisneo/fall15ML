#!/usr/bin/python

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB

###############################################################################


wordlist = ["love", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNKNOWN"]


def transfer(fileDj, vocabulary):
    stemmer = stem.snowball.EnglishStemmer()
    inputfile = open(fileDj)
    valueVector = np.zeros(len(vocabulary))
    while(1):
        tempstring = inputfile.readline()
        if not tempstring:
            break
        value = np.array(tempstring.strip('\n').split(' '))
        for i in range(len(value)):
            stemedword = stemmer.stem(value[i])
            if stemedword in vocabulary:
                indx = vocabulary.index(stemedword)
#                    print value[i]
                valueVector[indx] += 1
            else:
                valueVector[-1] += 1 # the "UNKNOWN" charactor
    inputfile.close()
    return valueVector

    # return BOWDj



def buildFeatMatrix(folderpath):
    first = True
    Matrix = []
    for filename in os.listdir(folderpath):
        fullpath = os.path.join(folderpath, filename)
        if first:
            Matrix = transfer(fullpath, wordlist)
            first = False
        else:
            temp = transfer(fullpath, wordlist)
            Matrix = np.vstack((Matrix, temp))
    return Matrix







def loadData(Path):
    if not os.path.isdir(path):
        print "invalid file path!"
        return
    traindir = os.path.join(path, "training_set")
    testdir = os.path.join(path, "test_set")
    trainPosDir = os.path.join(traindir, "pos")
    trainNegDir = os.path.join(traindir, "neg")
    testPosDir = os.path.join(testdir, "pos")
    testNegDir = os.path.join(testdir, "neg")
    # train
    print "building training pos"
    XtrainPos = naiveBayesMulFeature.buildFeatMatrix(self, trainPosDir)
    print "building training neg"
    XtrainNeg = naiveBayesMulFeature.buildFeatMatrix(self, trainNegDir)
    Xtrain = np.vstack((XtrainPos, XtrainNeg))
    ytrainPos = np.ones(len(XtrainPos))
    ytrainNeg = np.zeros(len(XtrainNeg))
    ytrain = np.hstack((ytrainPos, ytrainNeg))

    #test
    print "building test pos"
    XtestPos = naiveBayesMulFeature.buildFeatMatrix(self, testPosDir)
    print "building test neg"
    XtestNeg = naiveBayesMulFeature.buildFeatMatrix(self, testNegDir)
    Xtest = np.vstack((XtestPos, XtestNeg))
    ytestPos = np.ones(len(XtestPos))
    ytestNeg = np.zeros(len(XtestNeg))
    ytest = np.hstack((ytestPos, ytestNeg))
    
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):

    return Accuracy



def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):
    
    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):

    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    
    return yPredict, Accuracy


if __name__ == "__main__":

    # if len(sys.argv) != 3:
    #     print "Usage: python naiveBayes.py dataSetPath testSetPath"
    #     sys.exit()

    # print "--------------------"
    # textDataSetsDirectoryFullPath = sys.argv[1]
    # testFileDirectoryFullPath = sys.argv[2]


    # Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    # thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    # print "thetaPos =", thetaPos
    # print "thetaNeg =", thetaNeg
    # print "--------------------"

    # yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    # print "MNBC classification accuracy =", Accuracy

    # Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    # print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    # yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    # print "Directly MNBC tesing accuracy =", Accuracy
    # print "--------------------"

    # thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    # print "thetaPosTrue =", thetaPosTrue
    # print "thetaNegTrue =", thetaNegTrue
    # print "--------------------"

    # yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    # print "BNBC classification accuracy =", Accuracy
    # print "--------------------"

    path = "data_sets/training_set/pos/"

    M = buildFeatMatrix(path)
    print M
    print type(M)
    print len(M)