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
		
		labelMatrix = [] # label matrix, a list of ndarrays
		VADMatrix = [] # VAD matrix a list of ndarrays

		nameBegin = True # the begin indicator of the parsing of filenames

		# labelMatrixBegin = True # the begin indicator of the parsing of labelMatrix

		# VADMatrixBegin = True # the begin indicator of the parsing of VADMatrix

		catLabel = []
		# the entries in catLabel are :
		# 0: ground truth
		# middle part: evaluator category evaluations
		# -1: actor category evaluation

		VADLabel = []

		# 1D vector of VAD values : val, act, dom
		# 0 - 2 : ground truth
		# middle part : VAD evaluation of evaluators (all in 3 -tuples)
		# -3 - -1: actor VAD evaluation
		print labelPath
		count = -1
		while(1):

			count += 1

			tempstring = inputfile.readline()
			if not tempstring:
				if len(catLabel): # push the last instance of the file into matrix
					labelMatrix.append(catLabel)
				if len(VADLabel):
					VADMatrix.append(VADLabel)
				break
			tempstring = tempstring.strip()

			if not len(tempstring):
				# print len(tempstring)
				continue

			tempstring = tempstring.strip()
			

			# catGroundTruth = "" # category ground truth buffer
			
			if tempstring[0] == '[': # indicate the start of a new instance. read in the filename and the ground truth category label

				tempstring = tempstring.split('\t')

			# push the category label vector of previous instance into the matrix
				if len(catLabel):
					labelMatrix.append(catLabel)
				if len(VADLabel):
					VADMatrix.append(VADLabel)

				if nameBegin:
					filenames = np.asarray(tempstring[1])
					nameBegin = False
				else:
					filenames = np.hstack((filenames, np.asarray(tempstring[1])))
					# print 1
					# print filenames
				# print catLabel
				catLabel = np.asarray(tempstring[2])
				# print 2
				# print catLabel

				vadParse = tempstring[3].strip('[')
				vadParse = vadParse.strip(']')
				vadParse = vadParse.split(', ')
				# print vadParse

				VADLabel = np.asarray(float(vadParse[0]))
				# print VADLabel
				VADLabel = np.hstack((VADLabel, np.asarray(float(vadParse[1]))))
				# print VADLabel
				VADLabel = np.hstack((VADLabel, np.asarray(float(vadParse[2]))))
				# print VADLabel
	

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
					# print tempCate[6]
					for i in (1,3,-1):
						if tempCate[i].strip(';').strip().isdigit():
							VADLabel = np.hstack((VADLabel, float(tempCate[i].strip(';').strip())))
						else:
							VADLabel = np.hstack((VADLabel, -1))
			# readLabel.debug(self, VADLabel, "VADLabel")

			# elif labelMatrixBegin:
			# 	labelMatrix = catLabel
			# 	labelMatrixBegin = False

		inputfile.close()

		# print filenames
		# print filelabels
		# print type(labelMatrix)
		# print type(VADMatrix)
		labelMatrix = np.asarray(labelMatrix)
		VADMatrix = np.asarray(VADMatrix)
		# readLabel.debug(self,labelMatrix, "labelMatrix")
		# readLabel.debug(self,VADMatrix, "VADMatrix")
		# readLabel.debug(self,VADMatrix[0], "VADMatrix[0]")
		# readLabel.debug(self,labelMatrix[0], "labelMatrix[0]")
		return filenames, labelMatrix, VADMatrix

	def labelVectorGen(self, rangeStart, rangeEnd):
		fileNames = [] #fileNames of all files
		catLabels = [] #file cat labels of all files
		VADLabels = [] #file VAD labels of all files
		rootDir1 = "Session"
		Rdata = readLabel()
		gcount = 0

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
				name, label, vad = Rdata.readFile(fullpath)
				# print name
				# print "***********************************************"
				# if count == 0:
				# 	# fileNames = np.asarray(name).flatten()
				# 	# fileLabels = np.asarray(label)
				# 	# # fileLabels = label
				# 	# vadLabels = np.asarray(vad)


				# 	fileNames = list(name)
				# 	fileLabels = list(label)
				# 	vadLabels = list(vad)
				# 	# print fileNames
				# 	count = 1
				# 	# print type(fileNames)
				# else:
					# fileNames = np.hstack((fileNames, np.asarray(name).flatten()))

					# fileNames.append(list(name))
				name = list(name)
				# print name
				for i in name:
					# print i
					fileNames.append(i)
				# print "**************************"

				label = list(label)
				for i in label:
					catLabels.append(i)
				
				# # fileLabels.append(list(label))
				# for i in label:
				# 	fileLabels += i

				vad = list(vad)
				for i in vad:
					VADLabels.append(i)
					# # vadLabels.append(list(vad))
					# for i in vad:
					# 	vadLabels += i

					# readLabel.debug(self, vadLabels, "vadLabels")
		# readLabel.debug(self, fileNames, "fileNames")
		# readLabel.debug(self, fileLabels, "fileLabels")
		# readLabel.debug(self, vadLabels, "vadLabels")

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

		if os.path.isfile("names.txt"):
			os.remove("names.txt")
			os.remove("category.txt")
			os.remove("vad.txt")

		Rdata.saveToFile(fileNames, catLabels,VADLabels, ["names.txt", "category.txt", "vad.txt" ])
		return

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
	

	test = readLabel()
	# name, label, vad = test.readFile("Ses01F_impro01.txt")

	test.labelVectorGen(1, 2)
	# print vad[0]
	# test.logDebug(name, "name")
	# test.logDebug(label, "label")
	# test.logDebug(vad, "vad")



	# print len(label)
	# print len(vad)


		
		# print fileNames

		# print fileLabels

		# print len(fileNames)
		# print len(fileLabels)

	



