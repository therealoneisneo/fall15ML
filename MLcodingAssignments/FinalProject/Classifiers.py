# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:17:07 2015

@author: LZZ
"""
############################
## 1. Experiment Method, Import Data
#   expmethod = ["BasicAll", "All%", "Couple", "F&M"]
## 2. Adjust y Categories
#   catmethod = ["1-11", "1-8 other7", "XXXreplace", "XXXrpOther"]
## 3. Classifiers
#    3.1 Traditional Classifiers - see function for model choice
#   *3.2 Other Classifiers
## 4. Model Parameter Adjustment
#   Allow at most 2 paras adjusted for one model, otherwise seperated as new model
## 5. Cross Validation 
#   depend on experiment Method, loop or sklearn.cv
## 6. Print and Save results
# results, Experiment Method, Category method, clfdictory, adjpara1, adjpara2, Results(rates), clfmodel, other
############################

### initial settings
# 1 experiment method 
expmethod = ["BasicAll", "All%", "Couple", "F&M"]
# 2 categories method
catmethod = ["1-11", "1-8 other7", "XXXreplace", "XXXrpOther"]
# 3 define dictionary related to models
clfdict = {11: 'SVM linear',
           12: 'SVM rbf',
           13: 'SVM poly',
           21: 'log ovr',
           22: 'log multi',
           31: 'tree g r',
           32: 'tree e r',
           33: 'tree g b',
           34: 'tree e b',
           41: 'NB Gaussian',
           42: 'NB Multi',
           51: 'KNN uniform',
           52: 'KNN distance',
           }
# 6 define a black list for model result recordings

### import packages
## 1 import data Tdata
import numpy as np
import DataProcess as dp
Tdata = dp.trainingData()

## 3 import packagees for models
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes
from sklearn import neighbors
# 4 for cross validation
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

import time

# For Reference
# LabelDic = {'ang': 1, 'fru' : 2, 'sad' : 3, 'hap' : 4, 'neu' : 5, 'xxx' : 6 , 'sur' : 7, 'exc' : 8, 'dis' : 9, 'fea' : 10, 'oth': 11} # mapping of label to scalar category

############################
## Main Function Combine 1-6
############################

def mainfunction(expsnum, catsnum, models, adjpara1 = 1, adjpara2 = 1):
    ## 3 4 decide model clf; return clf
        clf = modelclf(models, adjpara1, adjpara2)
    ## 1 2 5 >>Do cross validation; return scores
        scores = crossvalidation(expsnum, catsnum, clf)
    ## 6 Record and Results
        print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2))
        other = ""
        newrecord = [np.mean(scores),expmethod[expsnum - 1], catmethod[catsnum - 1], clfdict[models], adjpara1, adjpara2, scores, clf, other]
        return newrecord


############################
## 3 4  Classifiers
############################

def modelclf(models, adjpara1 = 1, adjpara2 = 1):

    # Model - SVM - No.1X
    if models == 11:
        clf = svm.SVC(kernel = 'linear', C = adjpara1)
    elif models == 12:
        clf = svm.SVC(kernel = 'rbf', C = adjpara1)
    elif models == 13:
        clf = svm.SVC(kernel = 'poly', degree = adjpara1, gamma = adjpara2)
    # Models - Multivariate Logistic - No.2X     
    elif models == 21:
        clf = linear_model.LogisticRegression(penalty = adjpara2, C = adjpara1, multi_class = 'ovr')
    elif models == 22:
        clf = linear_model.LogisticRegression(penalty = adjpara2, C = adjpara1, multi_class = 'multinomial')
    # Models - Decision Tree - No.3X
    elif models == 31:
        clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = "random", max_features = adjpara1, max_depth = adjpara2)
    elif models == 32:
        clf = tree.DecisionTreeClassifier(criterion = "entropy", splitter = "random", max_features = adjpara1, max_depth = adjpara2)
    elif models == 33:
        clf = tree.DecisionTreeClassifier(criterion = "gini", splitter = "best", max_features = adjpara1, max_depth = adjpara2)
    elif models == 34:
        clf = tree.DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_features = adjpara1, max_depth = adjpara2)
    # Models - Naive Bayes - No.4X
    elif models == 41:
        clf = naive_bayes.GaussianNB()
    elif models == 42:
        clf = naive_bayes.MultinomialNB()
    # Models - KNN - No.5X
    elif models == 51:
        clf = neighbors.KNeighborsClassifier(n_neighbors = adjpara1, p = adjpara2, weights = 'uniform')
    elif models == 52:
        clf = neighbors.KNeighborsClassifier(n_neighbors = adjpara1, p = adjpara2, weights = 'distance')
    else:
        return
    
    return clf


############################
## 1 2 5 Import Data and change categories and Cross Validation
############################
## 2. Adjust y Categories
def adjcategories(trainy, catsnum):
    cat2 = {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8, 9:7, 10:7, 11:7}

    if catsnum == 2 or catsnum == 4:
        trainy = [cat2[y] for y in trainy]
    if catsnum == 1 or catsnum == 3:
        trainy = trainy
    return trainy

def crossvalidation(expsnum, catsnum, clf):
        
    if catsnum  == 1 or catsnum ==2:
        traindata =  "train.data"
    elif catsnum ==3 or catsnum == 4:
        traindata = "train1.data"
    
    if expsnum == 1:
        # import data
        InstanceDic, trainX, trainy = Tdata.getTrainingData(traindata, True)
        # adjust categories
        trainy = adjcategories(trainy, catsnum)
        # do cross validation by function and print scores directly
        scores = cross_validation.cross_val_score(clf, trainX, trainy, cv = 10)
    elif expsnum == 2:
        # import data
        InstanceDic, trainX, trainy = Tdata.getTrainingData(traindata, True)
        trainX, trainy, testX, testy = Tdata.controlCV(fold = 5) # the cross validation with session protion control, the argument "fold" in dicate the number of folds needed.
        # do cross validation by loop
        folds = range(5)
        scores = range(5)
        for fold in folds:
            trainXf = trainX[fold]
            trainyf = trainy[fold]
            testXf = testX[fold]
            testyf = testy[fold]
            # adjust categories
            trainyf = adjcategories(trainyf, catsnum)
            testyf =  adjcategories(testyf, catsnum)
            # now do models and scores
            clf.fit(trainXf, trainyf)
            y_pred = clf.predict(testXf)
            scores[fold] = accuracy_score(testyf, y_pred)
    elif expsnum == 3:
        folds = range(5)
        scores = range(5)
        for fold in folds:
            # import data
            InstanceDic, trainX, trainy = Tdata.getTrainingData(traindata, True)
            trainX, trainy, testX, testy = Tdata.leave1SesOut(leaveOutSes = fold + 1) # leave one session(couple) out, the argument is chose from 1 to 5
            # adjust categories
            trainy = adjcategories(trainy, catsnum)
            testy = adjcategories(testy, catsnum)
            # now do models and scores
            clf.fit(trainX, trainy)
            y_pred = clf.predict(testX)
            scores[fold] = accuracy_score(testy, y_pred)
    elif expsnum == 4:
        genders = ["F", "M"]
        scores = range(2)
        for gender in range(2):
            # import data
            InstanceDic, trainX, trainy = Tdata.getTrainingData(traindata, True)
            trainX, trainy, testX, testy = Tdata.male_female(traingender = genders[gender]) # split female and male samples, "traingender" can be "F" or "M", which determines the content in trainX&trainy
            # adjust categories
            trainy = adjcategories(trainy, catsnum)
            testy = adjcategories(testy, catsnum)
            # now do models and scores
            clf.fit(trainX, trainy)
            y_pred = clf.predict(testX)
            scores[gender] = accuracy_score(testy, y_pred)
    else:
        return
    
    return scores


############################
## Now do grid try
############################

def gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, recordload, recordsave):
    modelrecord = np.load(recordload)
    for expsnum in expsnums:
        for catsnum in catsnums:
            for models in modelss:
                for adjpara1 in adjpara1s:
                    for adjpara2 in adjpara2s:
                        start_time = time.clock()
                        try:
                            newrecord = mainfunction(expsnum, catsnum, models, adjpara1, adjpara2)
                        except:
                            print "error:", expsnum, catsnum, models, adjpara1, adjpara2
                            pass
                        modelrecord = np.vstack([modelrecord, newrecord])
                        print time.clock() - start_time, "seconds"
    
    np.save(recordsave, modelrecord)


############################
## TryTryTry Results
############################

## check if can pass error
#expsnums = [1]
#catsnums = [1]
#modelss = [13,42,51]
#adjpara1s = [1]
#adjpara2s = [1]
#gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "modelrecord626.npy", "modelrecord100.npy")

## check if can run for catsnum = 3 or 4
#expsnums = [3]
#catsnums = [3]
#modelss = [41]
#adjpara1s = [1]
#adjpara2s = [1]
#gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "modelrecord100.npy", "modelrecord120.npy")

#expsnums = [2]
#catsnums = [4]
#modelss = [41]
#adjpara1s = [1]
#adjpara2s = [1]
#gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "modelrecord120.npy", "modelrecord140.npy")

#expsnums = [1,2,3,4]
#catsnums = [1,2,3,4]
#modelss = [41]
#adjpara1s = [1]
#adjpara2s = [1]
#gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "modelrecord140.npy", "modelrecord150.npy")
# cat = 3 uniformly better result

# try on results
expsnums = [2]
catsnums = [4]
modelss = [34]
adjpara1s = [1]
adjpara2s = [1]
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "recordblank1.npy")


## For model 11, 12
expsnums = [1,2,3,4]
catsnums = [1,2,3,4]
modelss = [11,12]
adjpara1s = [0.01,0.1,0.5,1,5,10,100]
adjpara2s = [1]
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "record1112.npy")

## For model 13
expsnums = [1,2,3,4]
catsnums = [1,2,3,4]
modelss = [13]
adjpara1s = [0.01,0.1,0.5,1,5,10,100]
adjpara2s = [2,3,4,5]
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "record13.npy")

## For model 21
expsnums = [1,2,3,4]
catsnums = [1,2,3,4]
modelss = [21]
adjpara1s = [0.01,0.1,0.5,1,5,10,100]
adjpara2s = ['l1','l2']
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "record21.npy")

## For model 31, 32, 33, 34
expsnums = [1,2,3,4]
catsnums = [1,2,3,4]
modelss = [31, 32, 33, 34]
adjpara1s = [None, "sqrt", "log2"]
adjpara2s = [None, 10, 20, 30]
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "record3X.npy")

## For model 41
expsnums = [1,2,3,4]
catsnums = [1,2,3,4]
modelss = [41]
adjpara1s = [1]
adjpara2s = [1]
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "record41.npy")

## For model 51, 52
expsnums = [1,2,3,4]
catsnums = [1,2,3,4]
modelss = [51, 52]
adjpara1s = [3, 5, 10, 15, 20, 50, 100]
adjpara2s = [1, 2]
gridtry(expsnums, catsnums, modelss, adjpara1s, adjpara2s, "recordblank.npy", "record5152.npy")
