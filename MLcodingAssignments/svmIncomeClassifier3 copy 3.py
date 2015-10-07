import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
import sklearn.preprocessing as sp
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

    


class svmIncomeClassifier:
#    "the class for ML SVM coding assignment"
#    xVal = None # x values, the input value
#    yVal = None # y values, the target value
#    theta = None # theta value
    svm = 0

    def __init__(self):
        svm = 1
        
    def loadDataSet(self, filename):
        inputfile = open(filename)
        valueMatrix = []
        while(1):
            tempstring = inputfile.readline()
            
            if tempstring.isspace() or tempstring == "": 
                break
            
#            temp = instances(self, value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9], value[10], value[11], value[12], value[13], value[14])
            # tempvalue = []
            # for i in range(len(value)):
            #     tempvalue.append(float(value[i]))
            # value = np.array(tempvalue)
            value = np.array(tempstring.strip('\n').split(', '))
            valueMatrix.append(value)
        inputfile.close()   
        return valueMatrix


    def meanValue(self, data):
        sums = 0.0
        count = 0.0
        for i in range(len(data)):
            if data[i] != '?':
                sums = sums + float(data[i])
                count = count + 1
        return sums/count

    def processDataSet(self, train, test):
        print("Loading training data...")
        trainingData = svmIncomeClassifier.loadDataSet(self, train)
        print("Loading test data...")
        testData = svmIncomeClassifier.loadDataSet(self, test)
        trainNum = len(trainingData)
        testNum = len(testData)
        attributesNum = len(trainingData[0])
#        testNum = len(testData)

        print("processing data...")
        attributes = []
        
        for i in range(attributesNum):
            temp = [trainingData[0][i]]
            attributes.append(temp)
        
        
        for i in range(1, trainNum):
            for j in range(attributesNum):
                if attributes[j].count(trainingData[i][j]) == 0 and trainingData[i][j] != '?':
                    attributes[j].append(trainingData[i][j])
        for i in range(trainNum):
            for j in range(attributesNum):
                if j in [0, 2, 4, 10, 11, 12]:
                    if trainingData[i][j] != '?':
                        trainingData[i][j] = float(trainingData[i][j])
                    else:
                        trainingData[i][j] = "nan"
                else:
                    if trainingData[i][j] != '?':
                        trainingData[i][j] = attributes[j].index(trainingData[i][j])
                    else:
                        trainingData[i][j] = "nan"

        for i in range(testNum):
            for j in range(attributesNum):
                if j in [0, 2, 4, 10, 11, 12]:
                    if testData[i][j] != '?':
                        testData[i][j] = float(testData[i][j])
                    else:
                        testData[i][j] = "nan"
                else:
                    if testData[i][j] != '?':
                        testData[i][j] = attributes[j].index(testData[i][j])
                    else:
                        testData[i][j] = "nan"

        trainingData = np.asarray(trainingData)
        testData = np.asarray(testData) 
        trainingDataT = trainingData.T
        print("SVM training...")


        trainX = trainingData[:, :-1]
        trainy = trainingData[:, -1]
        testX = testData[:, :-1]
        testy = testData[:, -1]
        imp = sp.Imputer(missing_values='NaN', strategy='mean', axis=0)
        trainX= imp.fit_transform(trainX)
        testX = imp.fit_transform(testX)
        scaler = sp.MinMaxScaler()    
        trainX = scaler.fit_transform(trainX)
        testX = scaler.fit_transform(testX)

        outfile = open("poly4.txt","w")


        for g in [ 5, 10, 50 , 100]:
            for c in [1,10,50,100,500,1000]:
                clf = SVC(kernel = 'poly', degree = 4, gamma = g, coef0 = 1)
                clf.fit(trainX, trainy)
                trainscore = str(clf.score(trainX, trainy))
                testscore = str(clf.score(testX, testy))
                print("C = " + str(c) + ", kernel = poly , gamma = " + str(g) + ", degree = 4 , train score :  " + trainscrore + ", test score: " + testscore)
                outfile.write("C = " + str(c) + ", kernel = poly , gamma = " + str(g) + ", degree = 4 , train score :  " + trainscore + ", test score: " + testscore + '\n')



        # result = []
        # clf = SVC()
        # clf.fit(trainX, trainy)
        # print(clf.score(testX, testy))
        # print("checking result...")
        # for i in range(testNum):
        #     temp = clf.predict(testData[i])
        #     result.append(np.logical_xor(temp, testy[i]))
        outfile.close()

        return clf.score(testX, testy)
        
        
if __name__ == "__main__":

    svmC = svmIncomeClassifier()
    testresult = svmC.processDataSet("adult.data", "adult.test")
    print(testresult)
    a = 0 
    
    
    
    
    