#!/usr/bin/env python


# ML HW5
# jinlong Feng-jf5rb


import sys
import os
import copy as c
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
        if count[i] == 0:
            newseeds[i] = sums[i]
        else:
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
    print "k = " + str(k)
    seeds = seed(X, k)
    count = 0
    while (count < maxIter):
        count += 1
        # print "iteration: " + str(count) 
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
        # print diffsum
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
    colors = ["red", "blue", "yellow",  "green", "black", "orange", "purple"]

    plt.title("Scatter plot of clusters")
    if len(clusters) == 2:
        for i in range(k):
        # for i in [1]:
            plt.scatter(clusters[i].T[0], clusters[i].T[1], 2, color = colors[i])
        plt.show()
    # print "seeds"
    # print seeds
    return labels, objectiveFunc

def purity(labels, trueLabels): #claculate the purity of cluster
    check = set(labels - 1)
    # print check
    clusternum = len(check)
    check = set(trueLabels - 1)
    trueclusternum = len(check)
    count = np.zeros(clusternum)
    countMatrix = np.zeros((clusternum, trueclusternum))
    puri = np.zeros(clusternum)
    # print countMatrix[0, 0]
    # print check


    for i in range(len(labels)):
        count[labels[i] - 1] += 1

    for i in range(len(labels)):
        countMatrix[labels[i] - 1, trueLabels[i] - 1] += 1
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

def gmmCluster(X, k, covType, maxIter): # the guassian mixture model clustering
    print "GMM processing"
    instances = X[:, :-1]
    trueLabels = X[:, -1]
    tolerance = 0.00000000001
    diff = 1000000
    count = 0
    cov = np.cov(instances.T)
    if covType == "diag":
        cov = np.diag(np.diag(cov))
    det_cov = np.linalg.det(cov)
    seeds = seed(X, k)
    ins_num = len(instances)
    dim = len(instances[0])
    prob_xi_muj = np.zeros((ins_num, k))
    Ezij = np.zeros((ins_num, k))
    prob_mu = np.zeros(k)
    prob_mu += 1/float(k)

    while(diff > tolerance and count < maxIter):
        count += 1
        # print "iterations: " + str(count)

        # update prob_xi_muj
        for i in range(ins_num):
            for j in range(k):
                temp_dinom = det_cov
                # temp_dinom = temp_dinom * np.power((2 * np.pi), dim)
                temp_dinom = np.sqrt(temp_dinom)
                x_sub_mu = instances[i] - seeds[j]
                temp = np.dot(x_sub_mu, np.linalg.inv(cov))
                temp = np.dot(temp, x_sub_mu.T)
                temp = -0.5 * temp
                temp = np.exp(temp)
                prob_xi_muj[i, j] = float(temp) / float(temp_dinom)



        #update Ezij (E-step)
        for i in range(ins_num):
            denom = sum(prob_xi_muj[i] * prob_mu)
            for j in range(k): 
                Ezij[i, j] = prob_xi_muj[i, j] * prob_mu[j] / denom


        #update muj and prob_mu, M-step

        # print instances.shape
        # print Ezij.shape
        tempseeds = c.copy(seeds)

        for i in range(k):
            nom = 0
            denom = 0
            for j in range(ins_num):
                # print Ezij.T[i, j] #* instances
                nom += Ezij.T[i, j] * instances[j]
                denom += Ezij.T[i, j]
            # print nom
            # print denom
            seeds[i] = nom/denom
        # print seeds
        # print tempseeds

        diff = np.linalg.norm(seeds - tempseeds)
    print "total iterations :" + str(count)
    print "final diff : " + str(diff)
    # print Ezij
    labels = np.zeros(ins_num)
    for i in range(ins_num):
        labels[i] = np.argsort(Ezij[i])[-1] + 1
    # print labels
    clusters = []
    for i in range(k):
        clusters.append([])
    for i in range(len(labels)):
        clusters[int(labels[i] - 1)].append(instances[i])
    for i in range(len(clusters)):
        clusters[i] = np.asarray(clusters[i])

    colors = ["red","blue", "yellow",  "green", "black", "orange", "purple"]

    plt.title("Scatter plot of clusters")
    for i in range(k):
        plt.scatter(clusters[i].T[0], clusters[i].T[1], 2, color = colors[i])
    plt.show()

    




    # print cov.shape

    return labels




if __name__ == '__main__':
    path = sys.argv[1]
    # path = "data_sets_clustering"
    

    path1 = os.path.join(path, "humanData.txt")
    path2 =  os.path.join(path, "audioData.txt")


   


###########################   Task 1 K-means

    X = LoadData(path1)
    trueLabels = X.T[-1]
    labels = []
    pure = []
    obj = []
##################### Objctive funciton and purity

    for i in range(1, 7):
    # for i in range(2, 3):
        temp, objfunc = kmeans(X, i, 2000)
        obj.append(objfunc)
        labels.append(temp)
        pure.append(purity(temp,trueLabels))
    print "The objective function under k from 1 to 6 are: "
    print obj
    print "The cluster purity under k from 1 to 6 are: "
    print pure

##################### Plot only for path 1
    xaxis = range(1,7)

    plt.title("plot of objective function")
    
    plt.plot(xaxis, obj, color = "red", linewidth = 2)
    plt.show()
    


# ###########################   Task 2 GMM

    X = LoadData(path1)
    trueLabels = X.T[-1]
    labels = gmmCluster(X, 2, "full", 130)
    print "purity of GMM on dataset1 with full cov = " 
    print purity(labels, trueLabels)

    labels = gmmCluster(X, 2, "diag", 130)
    print "purity of GMM on dataset1 with diag cov = " 
    print purity(labels, trueLabels)

    X = LoadData(path2)
    trueLabels = X.T[-1]
    labels = gmmCluster(X, 2, "diag", 130)
    print "purity of GMM on dataset2 with diag cov = " 
    print purity(labels, trueLabels)





