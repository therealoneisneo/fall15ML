import numpy as np
import DataProcess as dp




Tdata = dp.trainingData()



# InstanceDic, trainX, trainy = Tdata.getTrainingData("train.data", False)

Tdata.getTrainingData("train.data", True) # get data from train.data

trainX, trainy, testX, testy = Tdata.leave1SesOut(leaveOutSes = 1) # leave one session(couple) out, the argument is chose from 1 to 5

trainX, trainy, testX, testy = Tdata.male_female(traingender = "F") # split female and male samples, "traingender" can be "F" or "M", which determines the content in trainX&trainy

trainX, trainy, testX, testy = Tdata.controlCV(fold = 5) # the cross validation with session protion control, the argument "fold" in dicate the number of folds needed.






