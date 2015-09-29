import numpy as np
import random as rd
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class ridgeRegression:
    "the class for ML ridgeRegression coding assignment"
    xVal = None # x values, the input value
    yVal = None # y values, the target value
    theta = None # theta value

    def __init__(self):
        self.xVal = None
        self.yVal = None
        self.theta = None
        self.dimension = None # the dimensionality of x


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
        # plt.title("Scatter plot of input data")
        # plt.scatter(self.xVal.T[1], self.yVal.T, 2)
        # plt.show()
        return

    def ridgeRegress(self, xVal, yVal, lamda = 0):
        xVal = np.asarray(xVal, float)
        yVal = np.asarray(yVal, float)

        xValT = xVal.T
        lambdaI = np.identity(self.dimension) * lamda
        temp = np.linalg.linalg.dot(xValT, xVal)
        temp = temp + lambdaI
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




    def cv(self, xVal, yVal):
        xVal = np.asarray(xVal, float)
        yVal = np.asarray(yVal, float)

        lamda = 0.0
        fold = 10

        indexarray = np.arange(len(xVal))
        np.random.shuffle(indexarray)

        fold_select = np.array_split(indexarray, fold)

        betaData = []
        errorData = []
        minError = 1000000000
        minBeta = []
        optimalLamda = 0
        while(lamda <= 1):
            sumError = 0
            for i in range(fold):
                test = fold_select[i]
                training = []
                for j in range(fold):
                    if j != i:
                        training = np.hstack((training, fold_select[j]))

                foldXVal = xVal[training[0]]
                foldYVal = yVal[training[0]]
                for j in range(1, len(training)):
                    foldXVal = np.vstack((foldXVal, xVal[training[j]]))
                    foldYVal = np.vstack((foldYVal, yVal[training[j]]))
                beta = ridgeRegression.ridgeRegress(self, foldXVal, foldYVal, lamda).T
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
                sumError = sumError + error
                if i == 0:
                    errorData = error
                else:
                    errorData = np.vstack((errorData, error))
            if sumError < minError:
                minError = sumError
                optimalLamda = lamda
            lamda = lamda + 0.02
        return optimalLamda


    def showRegPlot(self, theta = None):
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if theta == None:
            theta = self.theta
#         x1t = self.xVal.T[1]
#         x2t = self.xVal.T[2]
#         yt =  self.yVal.T
#         order = len(theta) - 1

#         y = theta[0]
#         for i in range(order):
#             y = y + theta[i + 1] * pow(x, i + 1)
#         ax.title("Result of linear regression")
#         ax.plot(x, y, color = "red",linewidth = 2)
        ax.scatter(self.xVal.T[1], self.xVal.T[2], self.yVal.T, c = 'r', marker = 'o')
        ax.plot_surface(self.xVal.T[1], self.xVal.T[2], self.yVal.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        x1 = np.linspace(-3.0, 3.0, 1000)
        x2 = np.linspace(-6.0, 6.0, 1000)
        y = self.theta[0] + self.theta[1] * x1 + self.theta[2] * x2
        ax.plot(x1, x2, y, label = "test")
#         ax.legend
        plt.show()
#         mpl.rcParams['legend.fontsize'] = 10
#
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
#         z = np.linspace(-2, 2, 100)
#         r = z**2 + 1
#         x = r * np.sin(theta)
#         y = r * np.cos(theta)
#         ax.plot(x, y, z, label='parametric curve')
#         ax.legend()
#
#         plt.show()



if __name__ == "__main__":

    rG = ridgeRegression()
    rG.loadDataSet("RRdata.txt")
    # theta = rG.ridgeRegress(rG.xVal, rG.yVal)
    bestlamda = rG.cv(rG.xVal, rG.yVal)
    rG.showRegPlot()





