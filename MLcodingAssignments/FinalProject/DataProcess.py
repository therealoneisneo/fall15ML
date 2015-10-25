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
		# 1 - 6 : C - E[1 - 6] category evaluation
		# 7: Female actor category evaluation
		# 8: male actoor category evaluation

		# VADlabel = [] 

		# 1D vector of VAD values : val, act, dom
		# 0 - 2 : ground truth
		# 3 - 20 : A-E[1-6] VAD evaluation
		# 21 - 23: Female actor VAD evaluation
		# 23 - 25: Male actor VAD evaluation

		
		labelMatrix = [] # label matrix
		VADMatrix = [] # VAD matrix

		nameBegin = True # the begin indicator of the parsing of filenames

		labelMatrixBegin = True # the begin indicator of the parsing of labelMatrix

		VADMatrixBegin = True # the begin indicator of the parsing of VADMatrix

		catLabel = []
		VADlabel = []

		while(1):
			tempstring = inputfile.readline()
			if not tempstring:
				break

			tempstring = tempstring.strip()
			tempstring = tempstring.split('\t')

			# catGroundTruth = "" # category ground truth buffer

			if tempstring[0][0] == '[': # read in the filename and the ground truth category label

			# initiate
				catLabelBegin = True
				VADlabelBegin = True # the begin indicator of this specific instance
				# catLabel = []
				# VADlabel = []
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
				VADlabel = np.asarray(float(vadParse[0]))
				VADlabel = np.hstack((VADlabel, np.asarray(float(vadParse[1]))))
				VADlabel = np.hstack((VADlabel, np.asarray(float(vadParse[2]))))

			else:
				if tempstring[0][0] == 'C':
					if tempstring[0][2] == 'E'
						cat = tempstring[1].strip(';')
						for i in range(1, 7):
							if int(tempstring[0][3]) == i:
								catLabel = np.hstack((catLabel, cat))
							else:
								catLabel = np.hstack((catLabel, "nan"))
					if tempstring[0][2] == 'F':

						

			elif labelMatrixBegin:
				labelMatrix = catLabel
				labelMatrixBegin = False

		inputfile.close()

		# print filenames
		# print filelabels
		return filenames, filelabels

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
	name, label = test.readFile("Ses01F_impro01.txt")
	print name
	print label


		
		# print fileNames

		# print fileLabels

		# print len(fileNames)
		# print len(fileLabels)

	



