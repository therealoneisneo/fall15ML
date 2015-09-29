import numpy as np
import random


indexarray1 = np.arange(20)
indexarray2 = np.arange(20)

np.random.shuffle(indexarray1)
random.shuffle(indexarray2)

print(indexarray1)
print(indexarray2)