import numpy as np
import os


class naiveBayesMulFeature:
    "the class for ML Bayes classifier coding assignment"
    
    def __init__(self):
        # self.xVal = None
        # self.yVal = None
        # self.theta = None
        # self.dimension = None # the dimensionality of x
        # self.optimalBeta = None
        # self.standBeta = None
        # self.instanceNum = 0
        self.wordnum = 18
        self.wordlist = ["love", "loving", "loves", "loved", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNKNOWN"]
        self.wordcounts = np.zeros(self.wordnum)
        self.wordcounts = np.ones(self.wordnum)

    def transfer(self, fileDj, vocabulary):
        inputfile = open(fileDj)
        valueVector = np.zeros(len(vocabulary))
        while(1):
            tempstring = inputfile.readline()
            if not tempstring:
                break
            value = np.array(tempstring.strip('\n').split(' '))
            for i in range(len(value)):
                if value[i] in vocabulary:
                    indx = vocabulary.index(value[i])
#                    print value[i]
                    valueVector[indx] += 1
                else:
                    valueVector[-1] += 1 # the "UNKNOWN" charactor
        inputfile.close()
        return naiveBayesMulFeature.wordlistProcess(self, valueVector)

    def wordlistProcess(self, wordcounts): # unifiy the count result of "love" words in the wordlist 
        # wordlist = np.delete(self.wordlist, 1)
        # wordlist = np.delete(self.wordlist, 1)
        # wordlist = np.delete(self.wordlist, 1)
        wordcounts[0] = wordcounts[0] + wordcounts[1] + wordcounts[2] + wordcounts[3]
        wordcounts = np.delete(wordcounts, 1)
        wordcounts = np.delete(wordcounts, 1)
        wordcounts = np.delete(wordcounts, 1)
        return wordcounts


    def buildFeatMatrix(self, folderpath):
        first = True
        Matrix = []
        for filename in os.listdir(folderpath):
            fullpath = os.path.join(folderpath, filename)
            if first:
                Matrix = naiveBayesMulFeature.transfer(self, fullpath, self.wordlist)
                first = False
            else:
                temp = naiveBayesMulFeature.transfer(self, fullpath, self.wordlist)
                Matrix = np.vstack((Matrix, temp))
        return Matrix


    def loadData(self, fullpath):
        # outputfile = open("test.txt", "w")
        if not os.path.isdir(fullpath):
            print "invalid file path!"
            return
        traindir = os.path.join(fullpath, "training_set")
        testdir = os.path.join(fullpath, "test_set")
        trainPosDir = os.path.join(traindir, "pos")
        trainNegDir = os.path.join(traindir, "neg")
        testPosDir = os.path.join(testdir, "pos")
        testNegDir = os.path.join(testdir, "neg")
        # train
        print "building training pos"
        XtrainPos = naiveBayesMulFeature.buildFeatMatrix(self, trainPosDir)
        print "building training neg"
        XtrainNeg = naiveBayesMulFeature.buildFeatMatrix(self, trainNegDir)
        Xtrain = np.vstack((XtrainPos, XtrainNeg))
        ytrainPos = np.ones(len(XtrainPos))
        ytrainNeg = np.zeros(len(XtrainNeg))
        ytrain = np.hstack((ytrainPos, ytrainNeg))

        #test
        print "building test pos"
        XtestPos = naiveBayesMulFeature.buildFeatMatrix(self, testPosDir)
        print "building test neg"
        XtestNeg = naiveBayesMulFeature.buildFeatMatrix(self, testNegDir)
        Xtest = np.vstack((XtestPos, XtestNeg))
        ytestPos = np.ones(len(XtestPos))
        ytestNeg = np.zeros(len(XtestNeg))
        ytest = np.hstack((ytestPos, ytestNeg))

        return Xtrain, Xtest, ytrain, ytest




    def train(self, Xtrain, ytrain):
        Vlen = len(Xtrain[0]) # the length of the vocabulary
        PosVocabularyNum = 0 # the # of words in the whole Pos
        NegVocabularyNum = 0 # the # of words in the whole Neg
        PosCategoryFrequency = np.zeros(Vlen) # the frequency of each words inside Pos
        NegCategoryFrequency = np.zeros(Vlen) # the frequency of each words inside Neg

        for i in range(len(Xtrain)):
            Wordcount = np.sum(Xtrain[i])
            if ytrain[i]: # a positive instance
                PosVocabularyNum += Wordcount
                for j in range(Vlen):
                    PosCategoryFrequency[j] += Xtrain[i,j]

                # print Wordcount
            else:
                NegVocabularyNum += Wordcount
                for j in range(Vlen):
                    NegCategoryFrequency[j] += Xtrain[i,j]
                # print Wordcount
        
        PosCategoryFrequency = (PosCategoryFrequencys) / PosVocabularyNum
        NegCategoryFrequncy = NegCategoryFrequency / NegVocabularyNum
        #test
        return


if __name__ == "__main__":
    nBF = naiveBayesMulFeature()
    a,b,c,d = nBF.loadData("data_sets")
    nBF.train(a,c)
    f = 1
    # TrainPosDir = "data_sets/training_set/pos/"
    # TrainNegDir = "data_sets/training_set/neg/"
    # TestPosDir = "data_sets/test_set/pos/"
    # TestNegDir = "data_sets/test_set/neg/"
    # for lists in os.listdir(rootDir): 
    #     path = os.path.join(rootDir, lists) 
    #     print path 
    # listdir = os.walk("data_sets/training_set/neg/")
    # print listdir
    # test = nBF.transfer("data_sets/training_set/neg/cv000_29416.txt", nBF.wordlist)
    # print test

    # nBF.wordlistProcess()
    # print nBF.wordnum
    # print nBF.wordlist
    # print nBF.wordcounts
