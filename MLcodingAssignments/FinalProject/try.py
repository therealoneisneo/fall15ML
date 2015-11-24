import numpy as np
import DataProcess as dp


a = range(15)


np.random.seed()

np.random.shuffle(a)

b = 6
if b in a:
	print "OK"


# print a 