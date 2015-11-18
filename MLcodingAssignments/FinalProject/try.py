import numpy as np
import DataProcess as dp

A = 3.0
d = 4.0


def con(A, d, limit):

	temp = (float(A)/float(d)) * np.sqrt(2/(np.pi * np.e))
	if temp < limit:
		return 1
	else:
		return 0 


print con(3,7,11)

