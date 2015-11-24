import numpy as np
import DataProcess as dp




Tdata = dp.trainingData()



# InstanceDic, trainX, trainy = Tdata.getTrainingData("train.data", False)

Tdata.getTrainingData("train.data", True)

a,b,c,d = Tdata.leave1SesOut(1)

# a,b,c,d = Tdata.male_female("F")

print a
print b
print c
print d


# for key in Tdata.InstanceDic:
# 	print len(Tdata.InstanceDic[key].featureVec)

