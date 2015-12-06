#!/usr/bin/env python


# ML HW5
# jinlong Feng-jf5rb


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def LoadData(filepath):   
    instancefile = open(filepath, 'r')
    rawparse = []

    print "processing data..."

    while(1):
        line = instancefile.readline()
        if not line:
            break

        line = line.strip().split(' ')
        rawparse.append(line)

    X = np.asarray(rawparse).astype(np.float)

    instancefile.close()
 
    return X

def seed(X, Snum): # generate Snum of initial cluster seeds for X input
    Seeds = []
    # print X.shape


    x = X.T
    dim = len(X[0]) - 1
    # print dim
    vrange = []
    # print x.shape

    for k in range(Snum):
        seed = []
        for i in range(dim):
            
            xmax = np.max(x[i])
            xmin = np.min(x[i])
            rand = np.random.random()
            temp = (xmax - xmin) * rand + xmin
            seed.append(temp)
        Seeds.append(seed)
    Seeds = np.asarray(Seeds).astype(np.float)
    return Seeds

def updateseed(X, labels, seeds):
    num = len(seeds)
    sums = np.zeros(seeds.shape)
    newseeds = np.zeros(seeds.shape)
    count = np.zeros(num)
    num = len(labels)
    for i in range(num):
        sums[labels[i]] += X[i]
        count[labels[i]] += 1
    # print sums
    # print count
    num = len(seeds)
    # print seeds
    for i in range(num):
        newseeds[i] = sums[i]/count[i]
    # print seeds

    return newseeds

def kmeans(X, k, maxIter):
    
    instances = X[:, :-1]
  
    num = len(instances)
    labels = np.zeros(num)
    tolerance = 0.00000001
    diff = 100000
    print "kmeans processing..."
    seeds = seed(X, k)
    count = 0
    while (count < maxIter):
        count += 1
        print "iteration: " + str(count)
        for i in range(num):
            dis = 100000000000
            for j in range(k):
                tempdis = np.linalg.norm(instances[i] - seeds[j])
                if tempdis < dis:
                    dis = tempdis
                    labels[i] = j
        newseeds = updateseed(instances, labels, seeds)
        diff = newseeds - seeds
        seeds = newseeds
        diffsum = 0
        for i in range(len(diff)):
            diffsum += np.linalg.norm(diff[i])
        if diffsum < tolerance:
            print diffsum
            break
        print diffsum
        # break
        # break  # or diff > tolerance

    # print seeds
    objectiveFunc = objFunc(instances, labels, seeds)

    clusters = []
    for i in range(k):
        clusters.append([])
    # print clusters
    for i in range(len(labels)):
        clusters[int(labels[i])].append(instances[i])
    for i in range(len(clusters)):
        clusters[i] = np.asarray(clusters[i])
    colors = ["blue", "yellow", "red", "green", "black", "orange", "purple"]

    # plt.title("Scatter plot of clusters")
    # for i in range(k):
    #     plt.scatter(clusters[i].T[0], clusters[i].T[1], 2, color = colors[i])
    # plt.show()
    return labels, objectiveFunc

def purity(labels, trueLabels): #claculate to purity of the cluster
    check = set(labels)
    # print check
    clusternum = len(check)
    check = set(trueLabels)
    trueclusternum = len(check)
    count = np.zeros(clusternum)
    countMatrix = np.zeros((clusternum, trueclusternum))
    puri = np.zeros(clusternum)
    # print countMatrix[0, 0]


    for i in range(len(labels)):
        count[labels[i]] += 1

    for i in range(len(labels)):
        countMatrix[labels[i], trueLabels[i] - 1] += 1
    # print countMatrix
    for i in range(clusternum):
        puri[i] = np.max(countMatrix[i])

    # mean = np.mean(puri/count)
    # print mean
    return sum(puri)/len(labels)



def objFunc(ins, labels, seeds):
    clu_num = len(seeds)
    ins_num = len(ins)
    obj = np.zeros(clu_num)
    # print len(labels)
    for i in range(ins_num):
        tempdis = np.linalg.norm(seeds[labels[i]] - ins[i])
        obj[labels[i]] += tempdis
        # break
    return sum(obj)

if __name__ == '__main__':
    # model = sys.argv[1]
    # train = sys.argv[2]
    # test = sys.argv[3]

    path1 = "data_sets_clustering/humanData.txt"
    path2 = "data_sets_clustering/audioData.txt"
    X = LoadData(path1)
    trueLabels = X.T[-1]
    labels = []
    pure = []
    obj = []

##################### Objctive funciton and purity

    for i in range(1, 7):
        temp, objfunc = kmeans(X, i, 2000)
        obj.append(objfunc)
        labels.append(temp)
        pure.append(purity(temp,trueLabels))
    print obj
    print pure

##################### Plot only for path 1
    xaxis = range(1,7)

    plt.title("plot of objective function")
    
    plt.plot(xaxis, obj, color = "red", linewidth = 2)
    plt.show()
    


    # # hard code for testing and debug

    # # model = "pcasvm"

    # # train = "zip.train"
    # # test = "zip.test"

    # # read_in_data(train , test)
    # xtrain, ytrain, xtest, ytest = read_in_data(train, test)

    # train = (xtrain, ytrain)
    # test = (xtest, ytest)

    # #----------------------------



    # if model == "dtree":
    #     print(decision_tree(train, test))
    # elif model == "knn":
    #     print(knn(train, test))
    # elif model == "net":
    #     print(neural_net(train, test))
    # elif model == "svm":
    #     print(svm(train, test))
    # elif model == "pcaknn":
    #     print(pca_knn(train, test))
    # elif model == "pcasvm":
    #     print(pca_svm(train, test))
    # else:
    #     print("Invalid method selected!")

    #-------------------------------------


    # test and experiment

    # combo_result = []
    # # combo_result.append("decision_tree")
    # # combo_result.append(decision_tree(train, test))
    # # combo_result.append("knn")
    # # combo_result.append(knn(train, test))
    # # combo_result.append("neural_net")
    # # combo_result.append(neural_net(train, test))
    # # combo_result.append("svm")
    # # combo_result.append(svm(train, test))
    # combo_result.append("pca_knn")
    # combo_result.append(pca_knn(train, test))
    # # combo_result.append("pca_svm")
    # # combo_result.append(pca_svm(train, test))

    # # outfile = open("result_combo.txt", 'w')
    # # for item in combo_result:
    # #     outfile.write(item)
    # #     outfile.write('\n')
    # # outfile.close()

    # print combo_result






