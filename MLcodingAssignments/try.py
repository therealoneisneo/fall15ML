import numpy as np
import random


indexarray1 = np.arange(30)
indexarray2 = np.arange(30)


#random.seed(1)
random.shuffle(indexarray1)
random.seed(37)
random.shuffle(indexarray2)

print(indexarray1)
print(indexarray2)