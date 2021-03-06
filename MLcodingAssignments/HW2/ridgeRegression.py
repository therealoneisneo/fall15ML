import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




class ridgeRegression:
    "the class for ML ridgeRegression coding assignment"
    
    def __init__(self):
        self.xVal = None
        self.yVal = None
        self.theta = None
        self.dimension = None # the dimensionality of x
        self.optimalBeta = None
        self.standBeta = None
        self.instanceNum = 0
        

    def loadDataSet(self, filename):
        inputfile = open(filename)
        valueMatrix = []
        while(1):
            tempstring = inputfile.readline()
            if tempstring == '':
                break
            value = np.array(tempstring.strip('\n').split(' '))
            tempvalue = []
            for i in range(len(value)):
                tempvalue.append(float(value[i]))
            value = np.array(tempvalue)
            valueMatrix.append(value)
        inputfile.close()
        
        valueMatrix = np.asarray(valueMatrix).T
        self.xVal = valueMatrix[:-1].T
        self.yVal = valueMatrix[-1:].T
        self.dimension = len(valueMatrix) - 1 
        self.instanceNum = len(valueMatrix.T)
        # in this ridge regression task, the constant entry is not considered at this stage
        # plt.title("Scatter plot of input data")
        # plt.scatter(self.xVal.T[1], self.yVal.T, 2)
        # plt.show()
        return
     
    def ridgeRegress(self, xVal, yVal, lamda = 0):
        xVal = np.asarray(xVal, float)
        yVal = np.asarray(yVal, float)
        # if lamda != 0:
        #     xValT = xVal.T[1:] # in this ridge regression task, the constant entry is not considered at this stage
        #     xVal = xValT.T
        #     lambdaI = np.identity(self.dimension) * lamda
        # else:
        xValT = xVal.T
        lambdaI = np.identity(self.dimension) * lamda
        temp = np.linalg.linalg.dot(xValT, xVal)
        temp = temp + lambdaI
        temp = np.linalg.inv(temp)
        temp = np.linalg.linalg.dot(temp, xValT)
        self.theta = np.linalg.linalg.dot(temp, yVal)
        if lamda == 0:
            self.standBeta = np.asarray(self.theta)
        return self.theta
    
    
    def standRegres(self, xVal, yVal):
        
        xVal = np.asarray(xVal, float)
        yVal = np.asarray(yVal, float)
        xValT = xVal.T
        temp = np.linalg.linalg.dot(xValT, xVal)
        temp = np.linalg.inv(temp)
        temp = np.linalg.linalg.dot(temp, xValT)
        self.theta = np.linalg.linalg.dot(temp, yVal)
        return self.theta
    
    def errorTest(self, beta, xVal, yVal):
        error = 0
        num = len(xVal)
        for i in range(num):
            temp = np.dot(beta, xVal[i])
            diff = np.abs(yVal[i] - temp)
            error = error + np.power(diff,2)
        return np.sqrt(error)
        # return error
    
#     def ridgeRegressCv(self, xVal, yVal, trainIndex, lamda = 0):
#         xVal = np.asarray(xVal, float)
#         yVal = np.asarray(yVal, float)
#         xValT = xVal.T
#         lambdaI = np.identity(self.dimension) * lamda
#         temp = np.linalg.linalg.dot(xValT, xVal)
#         temp = temp + lambdaI
#         temp = np.linalg.inv(temp)
#         temp = np.linalg.linalg.dot(temp, xValT)
#         self.theta = np.linalg.linalg.dot(temp, yVal)
#         return self.theta
    
    
#     def splitData(self, indexarray, testIndex, fold):#split the data into test set and training set in CV
#         
# #         xVal = np.asarray(xVal, float)
# #         yVal = np.asarray(yVal, float) 
# #         
#         training = None
#         test = None
#         instanceNum = len(indexarray)
#         testSize = int(instanceNum/fold)
#         teststart = testSize * testIndex
#         testend = min(teststart + testSize, instanceNum)
#         return (test, training)
        
        
#         
# for x in my_range(0, 1, 0.02):
#     print x
    
    
    def cv(self, xVal, yVal):
        xVal = np.asarray(xVal, float) 
        yVal = np.asarray(yVal, float)
      
        fold = 10
        
        # beta0 = 0

        # for i in range(self.instanceNum):
        #     beta0 = beta0 + yVal[i]
        # beta0 = np.asarray([[float(beta0)/float(self.instanceNum)]])

        # yVal = yVal - beta0
        
        indexarray = np.arange(len(xVal))
        random.seed(37)
        # random.shuffle(indexarray)
        
        # fold_select = np.array_split(indexarray, fold)
        
        betaData = []
        errorData = []
        minError = 1000000000
        optimalLamda = 0
        lamda = 0.0
        random.shuffle(indexarray)
        while(lamda <= 1):

            
            fold_select = np.array_split(indexarray, fold)

            beta = 0
            sumError = 0
            for i in range(fold): 
                test = fold_select[i]
                training = []
                for j in range(fold):
                    if j != i:
                        training = np.hstack((training, fold_select[j]))
                
    #             test, training = ridgeRegression.splitData(self, xVal, yVal, i, fold)
                
                foldXVal = xVal[training[0]]
                foldYVal = yVal[training[0]]
                for j in range(1, len(training)):
                    foldXVal = np.vstack((foldXVal, xVal[training[j]]))
                    foldYVal = np.vstack((foldYVal, yVal[training[j]]))

                beta = ridgeRegression.ridgeRegress(self, foldXVal, foldYVal, lamda).T
                if lamda == 0:
                    testbeta = ridgeRegression.standRegres(self, foldXVal, foldYVal)
                if i == 0:
                    betaData = beta
                else:
                    betaData = np.vstack((betaData, beta))
                testXVal = xVal[test[0]]
                testYVal = yVal[test[0]]
                for j in range(1, len(test)):
                    testXVal = np.vstack((testXVal, xVal[test[j]]))
                    testYVal = np.vstack((testYVal, yVal[test[j]]))
                error = ridgeRegression.errorTest(self, beta, testXVal, testYVal)
                sumError += error
            if lamda == 0:
                errorData = sumError
            else:
                errorData = np.vstack((errorData, sumError))
            print(str(lamda) + " : ")
            print(sumError)
            if sumError < minError:
                minError = sumError
                optimalLamda = lamda
                avbeta = 0
                for i in range(fold):
                    avbeta += betaData[i] 
                avbeta /= fold 
                self.optimalBeta = np.asarray(avbeta)
            lamda = lamda + 0.02

        # self.showCurvePlot(np.arange(0.02, 1, 0.02), errorData[1:])
        print ridgeRegression.ridgeRegress(self, xVal, yVal, optimalLamda)

        return optimalLamda


    def showCurvePlot(self, X, Y):
        plt.figure()
        plt.plot(X, Y)
        plt.show()
        
    def showRegPlot(self, theta = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(self.xVal.T[1], self.xVal.T[2], self.yVal.T[0], color = 'b', marker = 'o')
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y, sparse = True)
#         theta = np.asarray(theta[0])
#         print theta
     # cmap=cm.coolwarm,
# rstride=1, 0, antialiased=True linewidth=1
        Z = theta[0] + theta[1] * X + theta[2] * Y
        ax.plot_surface(X, Y, Z,  color = 'y')
        ax.set_zlim(-1.01, 1.01)

        plt.show()
        return
        
        
        
        
if __name__ == "__main__":

    rG = ridgeRegression()
    rG.loadDataSet("RRdata.txt")
    betaLR = rG.ridgeRegress(rG.xVal, rG.yVal)
    test = rG.standRegres(rG.xVal, rG.yVal)
    rG.showCurvePlot(rG.xVal.T[1], rG.xVal.T[2])
    print(betaLR)
    print(betaLR - test)
    rG.showRegPlot(rG.standBeta)
#     rG.showRegPlot(np.array((rG.standBeta[0,0],rG.standBeta[1,0],rG.standBeta[2,0])))
    bestlamda = rG.cv(rG.xVal, rG.yVal)
    print(bestlamda)
    rG.showRegPlot(rG.optimalBeta)
     
    
    
    
    
    