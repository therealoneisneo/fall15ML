import numpy as np
import DataProcess as dp

tempstring = "A-E4:	val 2; act 3; dom  3;	(guarded, tense, ready)"
# # 
tempstring = tempstring.strip()
tempstring = tempstring.split('\t')

print tempstring
a = tempstring[1]
a = a.split(' ')
print a 

for i in (1,3,5):
	print i

# for i in a:
# 	i = i.strip(';')
# 	print i
# print tempstring[2][1]

# parse = tempstring[3].strip('[')
# parse = parse.strip(']')
# parse = parse.split(', ')
# print tempstring[0][0:3]
# if tempstring[0][2] == 'E':
# 	print tempstring[0][3]

# cat = tempstring[1].strip(';')
# print cat
# print type(cat)
# print type(tempstring[3])
# print parse
# print type(parse[0])


