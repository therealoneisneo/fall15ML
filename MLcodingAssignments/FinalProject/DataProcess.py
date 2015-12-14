"""module for Data Process in ML fall2015 final project"""
__author__ = "jinlong Frank Feng"



import numpy as np
import wave as wv
import os
from features import mfcc
from features import logfbank
from features import fbank
import scipy.io.wavfile as wav

from pylab import*


mode = "full" # switch of the debug and full mode 
# mode = "debug"



LabelDic = {'ang': 1, 'fru' : 2, 'sad' : 3, 'hap' : 4, 'neu' : 5, 'xxx' : 6 , 'sur' : 7, 'exc' : 8, 'dis' : 9, 'fea' : 10, 'oth': 11} # mapping of label to scalar category


def debug(variable, varnamestr):
	print "Varname: " +  varnamestr  
	print "Type: " + str(type(variable))
	if type(variable) == 'numpy.ndarray':
		print "Length: " + str(variable.shape)
	else:
		print "Length: " + str(len(variable))
	print "Content: "
	if type(variable) == 'dict':
		for i in variable:
			print i
			print variable[i]
	else:
		print variable
	print " "
	return



def fdebug(variable, varnamestr):
	# readLabel.debug(self, variable, varnamestr)
	filename = varnamestr + ".dtxt"
	debuglog = open(filename, 'a')
	debuglog.write( "Varname: " +  varnamestr  + '\n')
	debuglog.write( "Type: " + str(type(variable)) + '\n')
	if type(variable) == 'numpy.ndarray':
		debuglog.write( "Length: " + str(variable.shape) + '\n')
	else:
		debuglog.write( "Length: " + str(len(variable)) + '\n')
	debuglog.write( "Content: " + '\n')

	if type(variable) == dict:
		for i in variable:
			print i
			print variable[i]
			debuglog.write(str(i) + ': \n')
			debuglog.write(str(variable[i]) + '\n')
		debuglog.write( " " + '\n')
	else:
		for i in variable:
			print i
			debuglog.write(str(i) + '\n')
		debuglog.write( " " + '\n')
	debuglog.close()

	return


class instances: # the class of a analyzed sentence


	def __init__(self):
		self.FileName = None # corresponding filename, which is a unique single name refering to a sentence, without .type. the related file could be .wav or .txt. 
		self.trainingLabel = None # the training label. which is just one emotion judgement generated by voting. For training purpose
		self.testLabel = [] # the label for accuracy test, could be a top x judgement.
		self.featureVec = [] # the composite feature vector for this instance, for training use
		# currently the structure of featureVec is :
		# 3-tuple(vad) + 12-tuple(mfcc) + 12-tuple(fbank) + TBC
		return
	def setValue(self, name, trainL, testL, Vec):
		self.FileName = name
		self.trainingLabel = trainL
		self.testLabel = testL
		self.featureVec = Vec
		return

	def copy(self, instances):
		self.setValue(instances.FileName, instances.trainL, instances.testL, instances.Vec)
		return

	def display(self): # show it self
		print self.FileName
		print self.trainingLabel
		print self.testLabel
		print self.featureVec
		return



class readAudio: # the class to read an audio file

	def __init__(self, audioPath = None):
		if audioPath != None:
			self.filePath = audioPath
		return


	def readFile(self, audioPath):
		self.filePath = audioPath

		return

	def featureDicGen(self, rangeStart, rangeEnd): # generate the dictionary of all instances as the key and their audio feature vector as value
		if mode == "full":
			rootDir = "fulldata"
		else:
			rootDir = ""
		rootDir1 = os.path.join(rootDir, "Session")
		featureDic = {}
		tempDiclist = [] # the list for append the tempfeatDic, after all calculations, add them into featureDic


		for i in range(rangeStart, rangeEnd + 1):
			dir1 = rootDir1 + str(i)
			dir1 = os.path.join(dir1, "sentences")
			if mode == "full":
				dir1 = os.path.join(dir1, "wav")
			else:
				dir1 = os.path.join(dir1, "wav(min)")
			for dir2 in os.listdir(dir1):
				dir3 = os.path.join(dir1, dir2)
				print dir3
				tempfeatDic = readAudio.dicGen(self, dir3)
				tempDiclist.append(tempfeatDic)

		

		for subDic in tempDiclist:
			for key in subDic:
				featureDic[key] = subDic[key]
		# fdebug(featureDic, "featureDic")

		return featureDic

	def dicGen(self, audioPath): # the sub routine in featureDicGen
		

		featureDic = {}
		# count = 0 # for test
		for item in os.listdir(audioPath):
			# count += 1

			# if count > 4:
			# 	break
			# debug(item,"item")
			if item[-4:] != ".wav":
				continue


			tempVec = [] # the feature vector of current clip being processed

			wavepath = os.path.join(audioPath, item)
			# print "processing " + wavepath
			rate, sig = wav.read(wavepath)
			
			mfcc_feat = mfcc(sig,rate)
			mfcc_feat = mfcc_feat[:, 1:13]
			# print mfcc_feat.shape
			# debug(mfcc_feat, "mfcc_feat")
			# debug(mfcc_feat[0], "mfcc_feat[0]")
			# break


			delta_mfcc = readAudio.deltacal(self, mfcc_feat)
			# print delta_mfcc.shape
			# debug(delta_mfcc, "delta_mfcc")
			# debug(delta_mfcc[0], "delta_mfcc[0]")
			# print type(delta_mfcc[0,0])
			# break


			deltadelta_mfcc = readAudio.deltadelta(self, delta_mfcc)
			# debug(deltadelta_mfcc, "deltadelta_mfcc")
			# print deltadelta_mfcc.shape
			# break

			mfcc_feat = np.mean(mfcc_feat, axis = 0)
			# print mfcc_feat.shape

			# break


			delta_mfcc = np.mean(delta_mfcc, axis = 0)
			deltadelta_mfcc = np.mean(deltadelta_mfcc, axis = 0)

			# take means of all windows in each dim for mfcc, delta_mfcc and deltadelta_mfcc


			fbank_feat, energy = fbank(sig,rate)
			# debug(energy, "energy")

			fbank_feat = np.mean(fbank_feat[:, 1:13], axis = 0)

			# debug(energy, "energy")
			
			energy_vec = []
			energyarray = np.asarray(energy)
			energy_vec.append(np.mean(energyarray, axis = 0))
			energy_vec.append(np.median(energyarray, axis = 0))
			energy_vec.append(np.std(energyarray, axis = 0))
			energy_vec.append(np.amax(energyarray, axis = 0))
			energy_vec.append(np.amin(energyarray, axis = 0))
			energy_vec = np.asarray(energy_vec)

			# debug(energy_vec, "test")

			# take the mean, dedian, standard diviation, max and min of the energy, 5 dim in all, into the feature vector

			# print len(delta_mfcc)
			tempVec.extend(mfcc_feat) # 12
			tempVec.extend(delta_mfcc)#12
			tempVec.extend(deltadelta_mfcc) #12
			tempVec.extend(fbank_feat) # 12
			tempVec.extend(energy_vec) # 5

			clip_name = item.split('.')[0] # name of the clip, as the key in the dictionary

			featureDic[clip_name] = tempVec
			# print len(tempVec)

			
			# break

		return featureDic

	def deltacal(self, win12array, N = 2):
		delta = []
		a = range(N + 1)
		denominate = 2 * np.dot(a, a)
		win12array = np.asarray(win12array).T
		row, col = win12array.shape
		window = range(-N, N + 1)
		for i in range(row):
			temprow = []
			for j in range(col - 2 * N):
				temprow.append(np.dot(win12array[i, j : j+5], window)/denominate)
			delta.append(np.asarray(temprow))
		delta = np.asarray(delta).T
		return delta
	    
	# second order
	def deltadelta(self, delta12array, N = 2):
	    return readAudio.deltacal(self, delta12array, N = 2)



class readLabel: # the class to read a label file

	def __init__(self, labelPath = None):
		self.filePath = labelPath
		return


	def readFile(self, labelPath = None): # read in a emotion evaluation file and return the filenames and the corresponding labels
		if labelPath != None:
			self.filePath = labelPath
		else:
			labelPath = self.filePath

		inputfile = open(self.filePath)

		instanceVec = []

		count = -1
		while(1):
			tempstring = inputfile.readline()
			if not tempstring:

				break
			
			tempstring = tempstring.strip()

			if not len(tempstring):
				continue

			if tempstring[0] == '[': # indicate the start of a new instance. read in the filename and the ground truth category label
				count += 1

				tempInstance = instances()

				tempstring = tempstring.split('\t')

				tempInstance.FileName = tempstring[1]
				
				tempInstance.trainingLabel = tempstring[2]

				vadParse = tempstring[3].strip('[')
				vadParse = vadParse.strip(']')
				vadParse = vadParse.split(', ')
				vadParse = [float(item) for item in vadParse] # the mean vad value from evaluators
				for item in vadParse:
					tempInstance.featureVec.append(item)

				instanceVec.append(tempInstance)




			else:
				if tempstring[0] == 'C': # A category label
					tempstring = tempstring.split('\t')
					tempCate = tempstring[1].split(' ')
					for cate in tempCate:
						cate = cate.strip(';')
						instanceVec[count].testLabel.append(cate)

		inputfile.close()


		# labelMatrix = np.asarray(labelMatrix)
		# VADMatrix = np.asarray(VADMatrix)

		# return filenames, labelMatrix, VADMatrix
		return instanceVec

	def labelVectorGen(self, rangeStart, rangeEnd):
		if mode == "full":
			rootDir = "fulldata"
		else:
			rootDir = ""
		rootDir1 = os.path.join(rootDir, "Session")
		# Rdata = readLabel()
		gcount = 0

		FullInsVec = []

		for i in range(rangeStart, rangeEnd + 1):
			dir1 = rootDir1 + str(i)
			dir1 = os.path.join(dir1, "dialog")
			dir1 = os.path.join(dir1, "EmoEvaluation")
			count = 0
			for files in os.listdir(dir1):
				if mode == "full":
					fullpath = os.path.join(dir1, files) # this is the operation code
				else:
					fullpath = os.path.join(dir1, "Ses01F_impro01.txt") #this is the simplified debug code
				if os.path.isfile(fullpath):
					
					# tempInsVec = Rdata.readFile(fullpath)
					tempInsVec = readLabel.readFile(self, fullpath)
					FullInsVec.extend(tempInsVec)
				# break # debug
		return FullInsVec

	def insVec2Dic(self, insVec):
		insDic = {}
		for item in insVec:
			insDic[item.FileName] = item
		return insDic

	def saveToFile(self, fileNames, catLabels, vadLabels, targetFileName):
		namefile = open(targetFileName[0], 'a')
		catfile = open(targetFileName[1], 'a')
 		vadfile = open(targetFileName[2], 'a')
		num = len(fileNames)
		for i in range(num):
			# write names
			temp = str(fileNames[i]) + '\n'
			namefile.write(temp)
			# write category
			for p in catLabels[i]:
				temp = str(p) + '\t'
				catfile.write(temp)
			catfile.write('\n')
			# write vad
		
			for p in vadLabels[i]:
				temp = str(p) + '\t'
				vadfile.write(temp)
			# vadfile.write(str(vadLabels[i]))
			vadfile.write('\n')

		namefile.close()	
		catfile.close()	
		vadfile.close()			
		return



class featureProcessing: # the class for the processing of instances features matrix

	def __init__(self):

		return

	def normalization(self, InstanceDic): # normalize the feature values of instances in the InstanceDic as (x - mu)/ std

		feat_matrix = []
		feat_matrix1 = []
		for ins in InstanceDic:
			feat_matrix.append(InstanceDic[ins].featureVec)
		# debug(feat_matrix, "feat_matrix")
		# print len(feat_matrix[0])
		meanvec = np.mean(feat_matrix, axis = 0)
		stdvec = np.std(feat_matrix, axis = 0)
		for item in feat_matrix:
			# debug(item, "item")
			item = np.subtract(item, meanvec)
			# debug(item, "item")
			item = np.true_divide(item, stdvec)
			# debug(item, "item")
			feat_matrix1.append(list(item))
			# break

		# debug(feat_matrix1, "feat_matrix1")
		
		

		count = 0
		for ins in InstanceDic:
			InstanceDic[ins].featureVec = feat_matrix1[count]
			InstanceDic[ins].trainingLabel = LabelDic[InstanceDic[ins].trainingLabel]
			count += 1

		return InstanceDic

	def getTrainData(self, InstanceDic, sort = False): # sort is the option sorting result
		trainX = []
		trainy = []
		name = []
		for key in InstanceDic:
			trainX.append(InstanceDic[key].featureVec)
			trainy.append(InstanceDic[key].trainingLabel)
			name.append(InstanceDic[key].FileName)
		if sort:
			alt_trainX = []
			alt_trainy = []
			alt_name = []
			sort_idx = np.argsort(name)

			for idx in sort_idx:
				alt_trainX.append(trainX[idx])
				alt_trainy.append(trainy[idx])
				alt_name.append(name[idx])
			# print alt_name
			return alt_trainX, alt_trainy, alt_name

		return trainX, trainy, name

	def writeFile(self, InstanceDic, DestFileName): # write all processed data to file
		outfile = open(DestFileName, 'w')
		for item in InstanceDic:
			outfile.write(str(InstanceDic[item].FileName) + '\n')
			outfile.write(str(InstanceDic[item].trainingLabel) + '\n')
			for p in InstanceDic[item].testLabel:
				outfile.write(str(p) + ' ')
			outfile.write('\n')
				# outfile.write(str(InstanceDic[item].testLabel) + '\n')
			for p in InstanceDic[item].featureVec:
				outfile.write(str(p) + ' ')
			outfile.write('\n')
			# outfile.write(str(InstanceDic[item].featureVec) + '\n')
		outfile.close()
		return

	def readInFile(self, featureFile): # read in processed feature from a file created by writeFile()

		InstanceDic = {}
		infile = open(featureFile, 'r')
		while(1):
			line = infile.readline()
			if not line:
				break
			if not len(line.strip()):
				continue
			if line[0:3] == 'Ses': # the start of an instance
				tempIns = instances()
				tempIns.FileName = line.strip() # store instance name
				tempIns.trainingLabel = int(infile.readline().strip())
				line = infile.readline().strip().split(' ') # read in test labels
				tempIns.testLabel = line
				line = infile.readline().strip().split(' ') # read in features
				tempIns.featureVec = [float(x) for x in line]

				InstanceDic[tempIns.FileName] = tempIns
		return InstanceDic


class trainingData:

	'''the class to get, store, split data for cross validation'''

	def __init__(self):
		self.InstanceDic = [] # the dictionary of all instance
		self.fulltrain = [] # the full training data , a tuple contains X and y

		return



	def getTrainingData(self, datafilename, processed = True): # the callable version of main. return the instance vector dictionary and trainX, trainy

		# processed = not processed

		# print processed
		fp = featureProcessing()

		if not processed:
			labels = readLabel()
			if mode == "full":
				InstanceVec = labels.labelVectorGen(1, 5)
			else:
				InstanceVec = labels.labelVectorGen(1, 1)
			# print type(InstanceVec)
			InstanceDic = labels.insVec2Dic(InstanceVec)


			audios = readAudio()
			if mode == "full":
				featureVecDic = audios.featureDicGen(1, 5)
			else:
				featureVecDic = audios.featureDicGen(1, 1)

			for item in InstanceDic:
				# print item
				InstanceDic[item].featureVec.extend(featureVecDic[item])



			
			InstanceDic = fp.normalization(InstanceDic)


			
			fp.writeFile(InstanceDic, datafilename)

		InstanceDic = fp.readInFile(datafilename)

		trainX, trainy, name = fp.getTrainData(InstanceDic)

		# debug(trainX, "trainX")
		# debug(trainy, "trainy")
		self.InstanceDic = InstanceDic
		self.fulltrain = trainX, trainy

		return InstanceDic, trainX, trainy


	def leave1SesOut(self, leaveOutSes = 0): # returns a pair of TrainX and Trainy, leave one session out for test, defined by "leaveOutSes"
		trainX = []
		trainy = []
		testX = []
		testy = []
		leaveOutSes = int(leaveOutSes)
		for key in self.InstanceDic:
			sesIdx = int(key.split('_')[0][-2])
			if sesIdx == leaveOutSes:
				testX.append(self.InstanceDic[key].featureVec)
				testy.append(self.InstanceDic[key].trainingLabel)
			else:
				trainX.append(self.InstanceDic[key].featureVec)
				trainy.append(self.InstanceDic[key].trainingLabel)

		return trainX, trainy, testX, testy

	def male_female(self, traingender = None): # split male and female instance and return training and testing data
		trainX = []
		trainy = []
		testX = []
		testy = []
		for key in self.InstanceDic:
			gender = key.split('_')[0][-1]
			if gender == traingender:
				trainX.append(self.InstanceDic[key].featureVec)
				trainy.append(self.InstanceDic[key].trainingLabel)
			else:
				testX.append(self.InstanceDic[key].featureVec)
				testy.append(self.InstanceDic[key].trainingLabel)
		return trainX, trainy, testX, testy 


	def controlCV(self, fold = 1): # cross validation data split, the portion of data extracted from each session is controled. returns train and test
	#train[i] and test[i] means the train and test data of i fold
		trainX = []
		trainy = []
		testX = []
		testy = []

		for i in range(fold):
			trainX.append([])
			trainy.append([])
			testX.append([])
			testy.append([])
		# print trainX
		# print trainy
		# print testX
		# print testy

		MaleTrainX = []
		FemaleTrainX = []
		Maletrainy = []
		Femaletrainy = []
		for i in range(5):
			MaleTrainX.append([])
			FemaleTrainX.append([])
			Maletrainy.append([])
			Femaletrainy.append([])
		# print MaleTrain
		# print FemaleTrain
		# print MaleTest
		# print FemaleTest

		for key in self.InstanceDic:

			gender = key.split('_')[0][-1]
			Ses = int(key.split('_')[0][-2]) - 1

			if gender == "F":
				FemaleTrainX[Ses].append(self.InstanceDic[key].featureVec)
				Femaletrainy[Ses].append(self.InstanceDic[key].trainingLabel)
			else:
				MaleTrainX[Ses].append(self.InstanceDic[key].featureVec)
				Maletrainy[Ses].append(self.InstanceDic[key].trainingLabel)
		# count = 0
		for i in range(5): # loop on Sessions
			np.random.seed()
			tempMaleidx = range(len(MaleTrainX[i]))
			tempFemaleidx = range(len(FemaleTrainX[i]))
			np.random.shuffle(tempMaleidx)
			np.random.shuffle(tempFemaleidx)

			Mfoldlength = len(tempMaleidx)/fold
			Ffoldlength = len(tempFemaleidx)/fold
			
			for j in range(fold): # loop on folds
				if j == fold - 1:
					Midxfold = tempMaleidx[j * Mfoldlength : ]
					Fidxfold = tempFemaleidx[j * Ffoldlength : ]
				else:
					Midxfold = tempMaleidx[j * Mfoldlength : (j + 1) * Mfoldlength]
					Fidxfold = tempFemaleidx[j * Ffoldlength : (j + 1) * Ffoldlength]
				for k in range(len(tempMaleidx)): # extract fold to train and test
					if k in Midxfold:# add to test
						testX[j].append(MaleTrainX[i][k])
						testy[j].append(Maletrainy[i][k])
					else:
						trainX[j].append(MaleTrainX[i][k])
						trainy[j].append(Maletrainy[i][k])
				for k in range(len(tempFemaleidx)): # extract fold to train and test
					if k in Fidxfold:# add to test
						testX[j].append(FemaleTrainX[i][k])
						testy[j].append(Femaletrainy[i][k])
					else:
						trainX[j].append(FemaleTrainX[i][k])
						trainy[j].append(Femaletrainy[i][k])
		# for i in range(fold):
		# 	print i
		# 	print len(trainX[i])
		# 	print len(trainy[i])
		# 	print len(testX[i])
		# 	print len(testy[i])
	
		return trainX, trainy, testX, testy
	def xxxtrans(self): # determing all instance with "xxx" label into its first emo evaluation
		Dic = self.InstanceDic
		# check = set()
		# count = 0
		for key in Dic:
			if Dic[key].trainingLabel == 6:
				# check.add(Dic[key].testLabel[0])
				temp = Dic[key].testLabel[0].lower()[:3]
				# print temp
				Dic[key].trainingLabel = LabelDic[temp]
				# print LabelDic[temp]
				# print Dic[key].trainingLabel
				# count += 1
		# print check
		return Dic

	def data4hmm(self): # transform the full data in to groups, each group is the sentenses from a single actor in a single conversation of a singel session. the returned groupDic is a dictionary can 
		Dic = self.InstanceDic
		groupDic = {} # the dictionary of lists, the key is the group's name, the value is a list of instance objects belong to the group
		groupNameSet = set()  # coresponding set to track the groups in the groupDic

		# form up groupDic
		for key in Dic:
			# groupname = Dic[key].FileName[:-3]
			tempname = Dic[key].FileName.split('_')
			groupname = tempname[0] + tempname[1] + tempname[-1][0]
			# print tgname
			# groupname = Dic[key].FileName.
			if groupname not in groupNameSet:
				temp = []
				temp.append(Dic[key])
				groupDic[groupname] = temp
				groupNameSet.add(groupname)
			else:
				groupDic[groupname].append(Dic[key])


		featureList = []

		fp = featureProcessing()


		# form up the featureList
		for key in groupDic:
			tempDic = {}
			for item in groupDic[key]:
				tempDic[item.FileName] = item
				# print item.FileName
			trainX, trainy, name = fp.getTrainData(tempDic, True)
			featureList.append((trainX, trainy, name))
		# for i in featureList:
		# 	fdebug(i[2], "featureList")
		
		return featureList

	

if __name__ == "__main__":
	
	Tdata = trainingData()
	Tdata.getTrainingData("train.data", True)
	# Dic = Tdata.xxxtrans();
	featureList = Tdata.data4hmm()

	# print len(Dic)

	# fp = featureProcessing()

	# fp.writeFile(Dic, "train1.data")
	
	# a,b,c,d = Tdata.leave1SesOut(2)

	# a,b,c,d = Tdata.male_female("F")
	# a,b,c,d = Tdata.controlCV(5)


	# print len(a[0])
	# print len(b[0])
	# print len(c[0])

	# print len(d[0])
	


