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

		filenames = []
		filelabels = []
		while(1):
			tempstring = inputfile.readline()
			if not tempstring:
				break
				tempstring.strip()
			if tempstring[0] == '[':
				tempstring = tempstring.split('\t')
				filenames.append(tempstring[1])
				filelabels.append(tempstring[2])
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

		Rdata.saveToFile(fileNames, fileLabels)
		return

	def saveToFile(self, fileNames, fileLabels):
		outfile = open("label_result.txt", 'w')
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

	



