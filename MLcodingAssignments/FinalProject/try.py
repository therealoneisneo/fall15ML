import numpy as np

a = "[6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]"

a = a.strip()

for i in a:
	if i == '\t':
		print i
		i = ' '
		print i
b = a.split(' ')



# print a
print b