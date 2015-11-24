import numpy as np
import DataProcess as dp


a = [1,2,3,4,5]
b = np.asarray(a)
c = list(b)
print type(a)
print a 

print type(b)
print b 


print type(c)
print c

a.extend(b)

print type(a)
print a