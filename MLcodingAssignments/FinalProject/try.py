import numpy as np
import DataProcess as dp

tempstring = "C-E2:	Neutral;	()"
# print tempstring
tempstring = tempstring.strip()
tempstring = tempstring.split('\t')

# parse = tempstring[3].strip('[')
# parse = parse.strip(']')
# parse = parse.split(', ')
# print tempstring[0][0:3]
if tempstring[0][2] == 'E':
	print tempstring[0][3]

# cat = tempstring[1].strip(';')
# print cat
# print type(cat)
# print type(tempstring[3])
# print parse
# print type(parse[0])
