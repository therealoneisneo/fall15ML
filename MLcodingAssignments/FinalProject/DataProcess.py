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

	def featureVecGen(self, audioPath):
		# mfcc_feat_vec = []
		# fbank_feat_vec = []
		featureVec = []
		for item in os.listdir(audioPath):
			tempVec = []

			wavepath = os.path.join(audioPath, item)
			rate, sig = wav.read(wavepath)
			
			mfcc_feat = mfcc(sig,rate)
			fbank_feat, enre = fbank(sig,rate) 
			print fbank_feat
			print len(fbank_feat)
			print fbank_feat[0]
			print enre
			print len(enre)
			# print mfcc_feat
			# print len(mfcc_feat)
			# print len(mfcc_feat[0])


			# meanMfcc = (np.sum(mfcc_feat, axis = 0)/len(mfcc_feat))[1:13] # the mean of the mfcc feature of all slide windows
			# meanFbank = (np.sum(fbank_feat, axis = 0)/len(mfcc_feat))[1:13]
			# tempVec = np.hstack((meanMfcc, meanFbank))
			# featureVec.append(tempVec)


			# print audiofeatureVec
			# print len(audiofeatureVec)

			# need a function here to process the multiple windows of mfcc and fbank in a single wave piece, here take the first window to process
			# mfccVec = mfcc_feat[0][1:13]
			# fbankVec = fbank_feat[0][1:13]
			# temp = np.hstack((mfccVec, fbankVec))
			# Vec.append(temp)

			# temp = np.fft.fft(sig)
		# print featureVec
		# print len(featureVec)
			# print np.linalg.norm(temp)
			break

		return

	#----------------------------------------------------------
	# the following functions in this class is from zhangzhang liu
	# delta is the approximate of 1 order derivitive

	def deltacal(win13array, N = 2):
	    n = np.arange(-N, N + 1)
	    denominate = sum(n**2)
	    numerator = [sum([win13array[j+i]*i for i in n]) for j in range(N,len(win13array) - N)]
	    return numerator/denominate
	    
	# second order
	def deltadelta(delta13array, N = 2):
	    return delta(delta13array, N = 2)



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

		# filenames = [] # 1D vector of filenames
		
		# labelMatrix = [] # label matrix, a list of ndarrays
		# VADMatrix = [] # VAD matrix a list of ndarrays

		# nameBegin = True # the begin indicator of the parsing of filenames

		# catLabel = []
		# # the entries in catLabel are :
		# # 0: ground truth
		# # middle part: evaluator category evaluations
		# # -1: actor category evaluation

		# VADLabel = []

		# # 1D vector of VAD values : val, act, dom
		# # 0 - 2 : ground truth
		# # middle part : VAD evaluation of evaluators (all in 3 -tuples)
		# # -3 - -1: actor VAD evaluation
		# print labelPath
		count = -1
		while(1):

			

			tempstring = inputfile.readline()
			if not tempstring:
				# if len(catLabel): # push the last instance of the file into matrix
				# 	labelMatrix.append(catLabel)
				# if len(VADLabel):
				# 	VADMatrix.append(VADLabel)
				break
			
			tempstring = tempstring.strip()

			if not len(tempstring):
				continue

			

			
			#------------------first version of matrix building (start)
			
			# if tempstring[0] == '[': # indicate the start of a new instance. read in the filename and the ground truth category label

			# 	tempstring = tempstring.split('\t')

			# # push the category label vector of previous instance into the matrix
			# 	if len(catLabel):
			# 		labelMatrix.append(catLabel)
			# 	if len(VADLabel):
			# 		VADMatrix.append(VADLabel)

			# 	if nameBegin:
			# 		filenames = np.asarray(tempstring[1])
			# 		nameBegin = False
			# 	else:
			# 		filenames = np.hstack((filenames, np.asarray(tempstring[1])))
			# 		# print 1
			# 		# print filenames
			# 	# print catLabel
			# 	catLabel = np.asarray(tempstring[2])
			# 	# print 2
			# 	# print catLabel

			# 	vadParse = tempstring[3].strip('[')
			# 	vadParse = vadParse.strip(']')
			# 	vadParse = vadParse.split(', ')
			# 	# print vadParse

			# 	VADLabel = np.asarray(float(vadParse[0]))
			# 	# print VADLabel
			# 	VADLabel = np.hstack((VADLabel, np.asarray(float(vadParse[1]))))
			# 	# print VADLabel
			# 	VADLabel = np.hstack((VADLabel, np.asarray(float(vadParse[2]))))
			# 	# print VADLabel

			#------------------first version of matrix building(end)
			if tempstring[0] == '[': # indicate the start of a new instance. read in the filename and the ground truth category label
				count += 1

				tempInstance = instances()

				tempstring = tempstring.split('\t')

				tempInstance.FileName = tempstring[1]
				
				tempInstance.trainingLabel = tempstring[2]

				vadParse = tempstring[3].strip('[')
				vadParse = vadParse.strip(']')
				vadParse = vadParse.split(', ')
				# print vadParse
				# vadParse = vadParse.astype(np.float)
				# vadParse = float(vadParse)
				# for item in vadParse:
				# for idx in range(len(vadParse)):
					# vadParse[i] = float(vadParse[i]
				# map(float, vadParse)
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

				# actor's self vad evaluation is not added in this version

				# if tempstring[0] == 'A-M' or tempstring[0] == 'A-F': # A VAD label
				# 	tempstring = tempstring.split('\t')
				# 	tempCate = tempstring[1].split(' ')
				# 	# print tempCate[6]
				# 	for i in (1,3,-1):
				# 		if tempCate[i].strip(';').strip().isdigit():
				# 			VADLabel = np.hstack((VADLabel, float(tempCate[i].strip(';').strip())))
				# 		else:
				# 			VADLabel = np.hstack((VADLabel, -1))


			# readLabel.debug(self, VADLabel, "VADLabel")

			# elif labelMatrixBegin:
			# 	labelMatrix = catLabel
			# 	labelMatrixBegin = False

		inputfile.close()


		# labelMatrix = np.asarray(labelMatrix)
		# VADMatrix = np.asarray(VADMatrix)

		# return filenames, labelMatrix, VADMatrix
		return instanceVec

	def labelVectorGen(self, rangeStart, rangeEnd):
		# fileNames = [] #fileNames of all files
		# catLabels = [] #file cat labels of all files
		# VADLabels = [] #file VAD labels of all files
		rootDir1 = "Session"
		Rdata = readLabel()
		gcount = 0

		FullInsVec = []

		for i in range(rangeStart, rangeEnd):
		# for i in range(1, 2):
			# fileNames = [] # fileNames of current session
			# fileLabels = [] # fileLabels of current session
			# vadLabels = [] # vadLabels of current session
			dir1 = rootDir1 + str(i)
			dir1 = os.path.join(dir1, "dialog")
			dir1 = os.path.join(dir1, "EmoEvaluation")
			count = 0
			for files in os.listdir(dir1):
				fullpath = os.path.join(dir1, files)
				if os.path.isfile(fullpath):
					
					tempInsVec = Rdata.readFile(fullpath)
					FullInsVec.extend(tempInsVec)
					

			# print len(FullInsVec)
			# for item in FullInsVec:
			# 	item.display()


		# debug filelabels line problem
		# a = 0
		# for k in fileLabels:
		# 	# for j in k:
		# 	if k[0] == 'xxx':
		# 		print k
		# 		a = k
		# 		break
		# readLabel.debug(self, a, "a")

		# debug filelabels line problem

		# print fileNames
		# print len(fileNames)

		# print len(filelabels)
		# print len(vadLabels)

		# if os.path.isfile("names.txt"):
		# 	os.remove("names.txt")
		# 	os.remove("category.txt")
		# 	os.remove("vad.txt")

		# Rdata.saveToFile(fileNames, catLabels,VADLabels, ["names.txt", "category.txt", "vad.txt" ])
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



	def fdebug(self, variable, varnamestr):
		# readLabel.debug(self, variable, varnamestr)
		debuglog = open("debuglog.txt", 'a')
		debuglog.write( "Varname: " +  varnamestr  + '\n')
		debuglog.write( "Type: " + str(type(variable)) + '\n')
		debuglog.write( "Length: " + str(len(variable)) + '\n')
		debuglog.write( "Content: " + '\n')
		for i in variable:
			print i
			debuglog.write(str(i) + '\n')
		debuglog.write( " " + '\n')
		debuglog.close()


	def debug(self, variable, varnamestr):
		print "Varname: " +  varnamestr  
		print "Type: " + str(type(variable))
		print "Length: " + str(len(variable))
		print "Content: "
		print variable
		print " "

if __name__ == "__main__":
	

	labels = readLabel()

	InstanceVec = labels.labelVectorGen(1, 2)
	InstanceDic = labels.insVec2Dic(InstanceVec)
	# print len(InstanceDic)
	# for item in InstanceDic:
	# 	InstanceDic[item].display()

	# InstanceVec and InstanceDic ready, waiting to put in feature info 
	
	
	# path = "Session1/sentences/wav/Ses01F_impro01"

	# audios = readAudio()
	# audios.featureVecGen(path)
	
	# path = os.path.join(path, "sentences")

	# for item in os.listdir(path):
	# 	wavepath = os.path.join(path, item)
	# 	rate, sig = wav.read(wavepath)
	# 	# output.write(str((float(len(b))/float(a))))
	# 	# output.write(str(a))
	# 	# output.write('\n')
	# 	# output.write(str(len(b)))
	# 	# output.write(str(b))
	# 	# output.write('\n')
	# 	mfcc_feat = mfcc(sig,rate)
	# 	fbank_feat = logfbank(sig,rate) 
	# 	# print mfcc_feat


	# 	# for item in fbank_feat:
	# 	# # for item in 
	# 	# 	print item
	# 	# 	break
	# 	print fbank_feat[0]
	# 	print mfcc_feat[0]
	# 	# for item in mfcc_feat:
	# 	# 	print item
	# 		# output.write(item)
	# 	# output.write(fbank_feat)
	# 	# print len(fbank_feat)
	# 	break
	# 	# print len(mfcc_feat)


		# temp = wv.open(wavepath, 'r')
		# print temp.getparams()
		# print temp.readframes(1000)

	# 	break
	# wavepath = os.path.join(path, item)
	# a, b = wavfile.read(path)
	# print a
	# print b


