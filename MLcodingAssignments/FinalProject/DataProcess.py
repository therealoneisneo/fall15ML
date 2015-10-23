import numpy as np
import wave as wv

class readAudio:

	def __init__(self, audioPath):
		self.filePath = audioPath
		return


	def readFile(self, audioPath):
		self.filePath = audioPath
		return


	def readFromFile(self):


class readLabel:

	def __init__(self, labelPath):
		self.filePath = labelPath
		return


	def readFile(self, labelPath):
		self.filePath = labelPath
		inputfile = open(labelPath)

		filenames = []
		filelables = []
		while(1):
            tempstring = inputfile.readline()
            if not tempstring:
                break
            if tempstring[0] == '[':
            	



        inputfile.close()



		print filenames
		print filelabels
		return


