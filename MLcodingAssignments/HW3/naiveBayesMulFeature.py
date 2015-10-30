import numpy as np
import os
from nltk import stem
from sklearn.naive_bayes import MultinomialNB



class naiveBayesMulFeature:
    "the class for ML Bayes classifier coding assignment"
    
    def __init__(self):

        # self.Vnum = 15
        # self.wordlist = ["love", "loving", "loves", "loved", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNKNOWN"]
        # self.ThetaPos = 0
        # self.ThetaNeg = 0

        self.Vnum = 15
        wordlist = ["love", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNKNOWN"]
        self.ThetaPos = 0
        self.ThetaNeg = 0
        stemmer = stem.snowball.EnglishStemmer()
        self.wordlist = []
        for word in wordlist:
            self.wordlist.append(stemmer.stem(word))

        # print self.wordlist


    def transfer(self, fileDj, vocabulary):
        stemmer = stem.snowball.EnglishStemmer()
        inputfile = open(fileDj)
        valueVector = np.zeros(len(vocabulary))
        while(1):
            tempstring = inputfile.readline()
            if not tempstring:
                break
            value = np.array(tempstring.strip('\n').split(' '))
            for i in range(len(value)):
                stemedword = stemmer.stem(value[i])
                if stemedword in vocabulary:
                    indx = vocabulary.index(stemedword)
#                    print value[i]
                    valueVector[indx] += 1
                else:
                    valueVector[-1] += 1 # the "UNKNOWN" charactor
        inputfile.close()
        return valueVector
        # return naiveBayesMulFeature.wordlistProcess(self, valueVector)

    def wordlistProcess(self, wordcounts): # unifiy the count result of "love" words in the wordlist 
        
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
        Vlen = self.Vnum # the length of the vocabulary
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
        for i in range(Vlen):
            PosCategoryFrequency[i] = (PosCategoryFrequency[i] + 1) / (PosVocabularyNum + Vlen)
            NegCategoryFrequency[i] = (NegCategoryFrequency[i] + 1) / (NegVocabularyNum + Vlen) 
        self.ThetaPos = np.log(PosCategoryFrequency)
        self.ThetaNeg = np.log(NegCategoryFrequency)
        return PosCategoryFrequency, NegCategoryFrequency


    def test(self, Xtest, ytest):
        TestNum = len(Xtest)
        yPredict = np.zeros(TestNum)
        ThetaPos = self.ThetaPos
        ThetaNeg = self.ThetaNeg
        for i in range(TestNum):
            pPos = np.dot(Xtest[i], ThetaPos)
            pNeg = np.dot(Xtest[i], ThetaNeg)
            # for j in range(self.Vnum): # the unknown words are not taken into consideration here
            #     pPos += Xtest[i, j] * ThetaPos[j]
            #     pNeg += Xtest[i, j] * ThetaNeg[j]
            if pPos > pNeg:
                yPredict[i] = 1
        Accuracy = 1 - float(np.sum(np.logical_xor(yPredict, ytest)))/TestNum
        return yPredict, Accuracy

    # def experiment(self, Xtrain):
    #     for i in range(len(Xtrain)):
    #         for j in range(len(Xtrain[i])):
    #             if Xtrain[i, j] > 0:
    #                 Xtrain[i, j] = 1
    #     return Xtrain   

    def sklearnComp(self, Xtrain, ytrain, Xtest, ytest):
        clf = MultinomialNB()
        clf.fit(Xtrain, ytrain)
        # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        # print(clf.predict(X[2]))

        return clf.score(Xtest, ytest)



if __name__ == "__main__":
    nBF = naiveBayesMulFeature()
    Xtrain, Xtest, ytrain, ytest = nBF.loadData("data_sets")
    # print Xtrain
    # Xtrain = nBF.experiment(Xtrain)
    # print Xtrain
    nBF.train(Xtrain,ytrain)
    print nBF.test(Xtest, ytest)
    print nBF.sklearnComp(Xtrain, ytrain, Xtest, ytest) 