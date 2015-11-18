import numpy as np
import DataProcess as dp



InstanceDic, trainX, trainy = dp.getTrainingData("traindata.dtxt", True)

dp.fdebug(InstanceDic, "InstanceDic")
dp.fdebug(trainX, "trainX")
dp.fdebug(trainy, "trainy")