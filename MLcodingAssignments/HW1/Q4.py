import numpy as np
import pylab
import matplotlib.pyplot as plt

class linearRegression:
    "the class for ML linearRegression coding assignment"
    xVal = None # x values, the input value
    yVal = None # y values, the target value
    theta = None # theta value

    def __init__(self):
        self.xVal = None
        self.yVal = None
        self.theta = None
        

    def loadDataSet(self, filename):
        inputfile = open(filename)
        valueMatrix = []
        while(1):
            tempstring = inputfile.readline()
            if tempstring == '':
                break
            value = np.array(tempstring.strip('\n').split('\t'))
            tempvalue = []
            for i in range(len(value)):
                tempvalue.append(float(value[i]))
            value = np.array(tempvalue)
            valueMatrix.append(value)
        inputfile.close()
        
        valueMatrix = np.asarray(valueMatrix).T
        self.xVal = valueMatrix[:-1].T
        self.yVal = valueMatrix[-1:].T
        plt.title("Scatter plot of input data")
        plt.scatter(self.xVal.T[1], self.yVal.T, 2)
        plt.show()
        return
     
    def standRegres(self, xVal = None, yVal = None):
        if xVal == None:
            xVal = self.xVal
        else:
            xVal = np.asarray(xVal, float)
        if yVal == None:
            yVal = self.yVal
        else:
            yVal = np.asarray(yVal, float)
        xValT = xVal.T
        temp = np.linalg.linalg.dot(xValT, xVal)
        temp = np.linalg.inv(temp)
        temp = np.linalg.linalg.dot(temp, xValT)
        self.theta = np.linalg.linalg.dot(temp, yVal)
        return self.theta
        
    def showRegPlot(self, theta = None):
        if theta == None:
            theta = self.theta
        order = len(theta) - 1
        x = np.linspace(0.0, 1.0, 1000)
        y = theta[0]
        for i in range(order):
            y = y + theta[i + 1] * pow(x, i + 1)
        plt.title("Result of linear regression")
        plt.plot(x, y, color = "red", linewidth = 2)
        plt.scatter(self.xVal.T[1], self.yVal.T, 2, color = "blue")
        plt.show()
        
        
        
if __name__ == "__main__":

    LG = linearRegression()
    LG.loadDataSet("Q4data.txt")
    theta = LG.standRegres()
    LG.showRegPlot()
    
    
    
    
    