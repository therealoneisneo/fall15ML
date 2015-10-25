"""module for Data Process in ML final project"""
__author__ = "jinlong Frank Feng"



import numpy as np
import wave as wv
import os

class readAudio: # the class to read an audio file

	def __init__(self, audioPath):
		self.filePath = audioPath
		return


	def readFile(self, audioPath):
		self.filePath = audioPath
		return



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

		filenames = [] # 1D vector of filenames
		# catLabel = [] # 1D vector of category label

		# the entries in catLabel are :
		# 0: ground truth
		# middle part: evaluator category evaluations
		# -1: actor category evaluation

		# VADLabel = [] 

		# 1D vector of VAD values : val, act, dom
		# 0 - 2 : ground truth
		# middle part : VAD evaluation of evaluators (all in 3 -tuples)
		# -3 - -1: actor VAD evaluation
		
		labelMatrix = [] # label matrix, a list of ndarrays
		VADMatrix = [] # VAD matrix a list of ndarrays

		nameBegin = True # the begin indicator of the parsing of filenames

		labelMatrixBegin = True # the begin indicator of the parsing of labelMatrix

		VADMatrixBegin = True # the begin indicator of the parsing of VADMatrix

		catLabel = []
		VADLabel = []

		while(1):
			tempstring = inputfile.readline()
			if not tempstring:
				break

			tempstring = tempstring.strip()
			

			# catGroundTruth = "" # category ground truth buffer
			
			if tempstring[0] == '[': # indicate the start of a new instance. read in the filename and the ground truth category label

				tempstring = tempstring.split('\t')

			# push the category label vector of previous instance into the matrix
				if catLabel:
					labelMatrix.append(catLabel)
				if VADLabel:
					VADMatrix.append(VADLabel)

			# initiate
				catLabelBegin = True
				VADLabelBegin = True # the begin indicator of this specific instance
				# catLabel = []
				# VADLabel = []
			# the begin of an instance
				
				if nameBegin:
					filenames = np.asarray(tempstring[1])
					nameBegin = False
				else:
					filenames = np.hstack((filenames, np.asarray(tempstring[1])))

				catLabel = np.asarray(tempstring[2])

				vadParse = tempstring[3].strip('[')
				vadParse = vadParse.strip(']')
				vadParse = vadParse.split(', ')
				VADLabel = np.asarray(float(vadParse[0]))
				VADLabel = np.hstack((VADLabel, np.asarray(float(vadParse[1]))))
				VADLabel = np.hstack((VADLabel, np.asarray(float(vadParse[2]))))

			else:
				if tempstring[0] == 'C': # A category label
					tempstring = tempstring.split('\t')
					tempCate = tempstring[1].split(' ')
					for cate in tempCate:
						cate = cate.strip(';')
						catLabel = np.hstack((catLabel, cate))

				if tempstring[0] == 'A': # A VAD label
					tempstring = tempstring.split('\t')
					tempCate = tempstring[1].split(' ')
					for i in (1,3,5):
						VADLabel = np.hstack((VADLabel, tempstring[1][i].strip(';')))

			# elif labelMatrixBegin:
			# 	labelMatrix = catLabel
			# 	labelMatrixBegin = False

		inputfile.close()

		# print filenames
		# print filelabels
		return filenames, labelMatrix, VADMatrix

	def labelVectorGen(self, rangeStart, rangeEnd):
		gfileNames = [] #fileNames of all files
		gfileLabels = [] #filelabels of all files
		rootDir1 = "Session"
		Rdata = readLabel()
		gcount = 0

		for i in range(rangeStart, rangeEnd + 1):
		# for i in range(1, 2):
			fileNames = [] # fileNames of current session
			fileLabels = [] # fileLabels of current session
			dir1 = rootDir1 + str(i)
			dir1 = os.path.join(dir1, "dialog")
			dir1 = os.path.join(dir1, "EmoEvaluation")
			count = 0
			for files in os.listdir(dir1):
				fullpath = os.path.join(dir1, files)
				name, label = Rdata.readFile(fullpath)
				if count == 0:
					fileNames = np.asarray(name).flatten()
					fileLabels = np.asarray(label).flatten()
					count = 1
				else:
					fileNames = np.hstack((fileNames, np.asarray(name).flatten()))
					fileLabels = np.hstack((fileLabels, np.asarray(label).flatten()))

			fileNames = np.asarray(fileNames)
			fileNames = fileNames.flatten()
			fileLabels = np.asarray(fileLabels)
			fileLabels = fileLabels.flatten()

			# if gcount == 0:
			# 	gfileNames = fileNames
			# 	gfileLabels = fileLabels
			# else:
			# 	gfileNames = np.hstack((gfileNames, fileNames))
			# 	gfileLabels


			#finishe this after the formulation of label matrix

		Rdata.saveToFile(fileNames, fileLabels, "label_result.txt")
		return

	def saveToFile(self, fileNames, fileLabels, targetFileName):
		outfile = open(targetFileName, 'w')
		num = len(fileNames)
		for i in range(num):
			temp = str(fileNames[i]) + '\t' + str(fileLabels[i]) + '\n'
			outfile.write(temp)
			# print temp
		outfile.close()			
		return



if __name__ == "__main__":
	

	test = readLabel()
	name, label, vad = test.readFile("Ses01F_impro01.txt")
	print name
	print label
	print vad


		
		# print fileNames

		# print fileLabels

		# print len(fileNames)
		# print len(fileLabels)

	



