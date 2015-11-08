#!/usr/bin/python

import sys
import os
import numpy as np
from nltk import stem
from sklearn.naive_bayes import MultinomialNB

###############################################################################



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
        Matrix.append(transfer(fullpath, wordlist))

        # if first:
        #     Matrix = transfer(fullpath, wordlist)
        #     first = False
        # else:
        #     temp = transfer(fullpath, wordlist)
        #     Matrix = np.vstack((Matrix, temp))
    # print Matrix
    return np.asarray(Matrix)


def loadData(Path):
    if not os.path.isdir(Path):
        print "invalid file path!"
        return
    traindir = os.path.join(Path, "training_set")
    testdir = os.path.join(Path, "test_set")
    trainPosDir = os.path.join(traindir, "pos")
    trainNegDir = os.path.join(traindir, "neg")
    testPosDir = os.path.join(testdir, "pos")
    testNegDir = os.path.join(testdir, "neg")
    # train
    print "building training pos"
    XtrainPos = buildFeatMatrix(trainPosDir)
    print "building training neg"
    XtrainNeg = buildFeatMatrix(trainNegDir)
    Xtrain = np.vstack((XtrainPos, XtrainNeg))
    ytrainPos = np.ones(len(XtrainPos))
    ytrainNeg = np.zeros(len(XtrainNeg))
    ytrain = np.hstack((ytrainPos, ytrainNeg))

    #test
    print "building test pos"
    XtestPos = buildFeatMatrix(testPosDir)
    print "building test neg"
    XtestNeg = buildFeatMatrix(testNegDir)
    Xtest = np.vstack((XtestPos, XtestNeg))
    ytestPos = np.ones(len(XtestPos))
    ytestNeg = np.zeros(len(XtestNeg))
    ytest = np.hstack((ytestPos, ytestNeg))
    
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    PosVocabularyNum = 0 # the # of words in the whole Pos
    NegVocabularyNum = 0 # the # of words in the whole Neg
    PosCategoryFrequency = np.zeros(Vlen) # the frequency of each words inside Pos
    NegCategoryFrequency = np.zeros(Vlen) # the frequency of each words inside Neg

    for i in range(len(Xtrain)):
        Wordcount = np.sum(Xtrain[i])
        if ytrain[i]: # a positive instance
            PosVocabularyNum += Wordcount
            for j in range(Vlen):
                PosCategoryFrequency[j] += Xtrain[i,j]

            # print Wordcount
        else:
            NegVocabularyNum += Wordcount
            for j in range(Vlen):
                NegCategoryFrequency[j] += Xtrain[i,j]
            # print Wordcount
    for i in range(Vlen):
        PosCategoryFrequency[i] = (PosCategoryFrequency[i] + 1) / (PosVocabularyNum + Vlen)
        NegCategoryFrequency[i] = (NegCategoryFrequency[i] + 1) / (NegVocabularyNum + Vlen) 

    thetaPos = PosCategoryFrequency
    thetaNeg = NegCategoryFrequency

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    # yPredict = []

    TestNum = len(Xtest)
    yPredict = np.zeros(TestNum)
    ThetaPos = np.log(thetaPos).T
    ThetaNeg = np.log(thetaNeg).T

    # print Xtest.shape
    # print thetaPos.shape 

    for i in range(TestNum):

        pPos = np.dot(Xtest[i], ThetaPos)
        pNeg = np.dot(Xtest[i], ThetaNeg)
        if pPos > pNeg:
            yPredict[i] = 1
    Accuracy = 1 - float(np.sum(np.logical_xor(yPredict, ytest)))/TestNum

    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)

    return clf.score(Xtest, ytest)



def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):

    stemmer = stem.snowball.EnglishStemmer()

    inputfile = open(path)

    posValue = 0
    negValue = 0

    while(1):
        tempstring = inputfile.readline()
        if not tempstring:
            break
        value = np.array(tempstring.strip('\n').split(' '))
        for i in range(len(value)):
            stemedword = stemmer.stem(value[i])
            if stemedword in wordlist:
                indx = wordlist.index(stemedword)
                posValue += thetaPos[indx]
                negValue += thetaNeg[indx]
            else:
                posValue += thetaPos[-1]
                negValue += thetaNeg[-1]
    if posValue >= negValue:
        yPredict = 1
    else:
        yPredict = 0

    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    thetaPos = np.log(thetaPos)
    thetaNeg = np.log(thetaNeg)
    total = 0.0
    correct = 0.0
    yPredict = []


    if not os.path.isdir(path):
        print "invalid file path!"
        return
    testPosDir = os.path.join(path, "pos")

    for filename in os.listdir(testPosDir):
        total += 1
        fullpath = os.path.join(testPosDir, filename)
        predict = naiveBayesMulFeature_testDirectOne(fullpath,thetaPos, thetaNeg)
        yPredict.append(predict)
        if predict == 1:
            correct += 1


    testNegDir = os.path.join(path, "neg")

    for filename in os.listdir(testNegDir):
        total += 1
        fullpath = os.path.join(testNegDir, filename)
        predict = naiveBayesMulFeature_testDirectOne(fullpath,thetaPos, thetaNeg)
        yPredict.append(predict)
        if predict == 0:
            correct += 1

    Accuracy = correct/total

    return yPredict, Accuracy

def Mul_to_Bern_train_tranfer(Xtrain):
    for i in range(len(Xtrain)):
        for j in range(len(Xtrain[0])):
            if Xtrain[i, j] > 0:
                Xtrain[i, j] = 1


    return Xtrain

def naiveBayesBernFeature_train(Xtrain, ytrain):

    PosNum = sum(ytrain) # the # of pos instances
    NegNum = len(ytrain) - PosNum # the # of neg instances

    posTextcount = np.zeros(Vlen) # the number of textfiles in pos category that contain a word in vocabulary
    negTextcount = np.zeros(Vlen) # the number of textfiles in neg category that contain a word in vocabulary

    Xtrain = Mul_to_Bern_train_tranfer(Xtrain)
    # print Xtrain

    for i in range(len(Xtrain)):
        # Wordcount = np.sum(Xtrain[i])
        if ytrain[i]: # a positive instance
            posTextcount += Xtrain[i]
        else:
            negTextcount += Xtrain[i]

    posTextcount += 1
    negTextcount += 1
    posTextcount = posTextcount/(PosNum + 2)
    negTextcount = negTextcount/(NegNum + 2)

    # self.ThetaPos = posTextcount
    # self.ThetaNeg = negTextcount
    return posTextcount, negTextcount

    # return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):

    TestNum = len(Xtest)
    yPredict = np.zeros(TestNum)
    ThetaPos = np.log(thetaPosTrue)
    ThetaNeg = np.log(thetaNegTrue)
    for i in range(TestNum):
        pPos = np.dot(Xtest[i], ThetaPos)
        pNeg = np.dot(Xtest[i], ThetaNeg)
        if pPos > pNeg:
            yPredict[i] = 1
    Accuracy = 1 - float(np.sum(np.logical_xor(yPredict, ytest)))/TestNum
    return yPredict, Accuracy
    

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()


    Owordlist = ["love", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNKNOWN"]
    wordlist = []

    stemmer = stem.snowball.EnglishStemmer()

    for word in Owordlist:
        wordlist.append(stemmer.stem(word))
    Vlen = len(wordlist)

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"



#--------------------------------------------------------------------------------------------
    


    # Owordlist = ["love", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNKNOWN"]
    # wordlist = []

    # stemmer = stem.snowball.EnglishStemmer()

    # for word in Owordlist:
    #     wordlist.append(stemmer.stem(word))
    # Vlen = len(wordlist)

    # path = "data_sets"#/training_set/pos/"

    # Xtrain, Xtest, ytrain, ytest = loadData(path)

    # thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)

    # # print thetaPos
    # # print thetaNeg

    # # testpath = "data_sets/test_set"
    # # yPredict, Accuracy = naiveBayesMulFeature _testDirect(testpath, thetaPos, thetaNeg)
    # # print "Directly MNBC tesing accuracy =", Accuracy

    # thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    # print "thetaPosTrue =", thetaPosTrue
    # print "thetaNegTrue =", thetaNegTrue
    # print "--------------------"

    # yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    # print "BNBC classification accuracy =", Accuracy
    # print "--------------------"
    # print yPredict

