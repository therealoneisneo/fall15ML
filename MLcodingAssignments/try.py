import numpy as np
import pylab
import matplotlib.pyplot as plt


test = np.array([[0, 1, 2]])

test2 = np.concatenate((test, [[1,2,9], [4,3,4]]))

test3 = test2.T

a = 0