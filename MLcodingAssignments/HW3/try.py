import numpy as np


a = np.ones(10)
b = np.zeros(5)
c = np.hstack((a, b))
c = np.asarray(c)
c = c.T
print c