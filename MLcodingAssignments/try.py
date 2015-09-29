import numpy as np


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
        
        
for x in my_range(0, 1, 0.02):
    print x

