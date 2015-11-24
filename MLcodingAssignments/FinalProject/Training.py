import numpy as np
import DataProcess as dp




Tdata = dp.trainingData()



InstanceDic, trainX, trainy = Tdata.getTrainingData("train.data", False)



for key in Tdata.InstanceDic:
	print len(Tdata.InstanceDic[key].featureVec)

