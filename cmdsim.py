#!/usr/bin/python

import shearex, sys
import numpy as np

a,b,c=np.float(sys.argv[1]),np.float(sys.argv[2]),np.float(sys.argv[3])
e=int(sys.argv[5])

if str.isdigit(sys.argv[4]):
    d=int(sys.argv[4])
elif str.isdigit(str.split(str.split(sys.argv[4],',')[0],'.')[0]):
    d=eval('[' + sys.argv[4] + ']')
else:
    d=sys.argv[4]

print a,b,c,d

shearex.simsky(a,b,c,d,e)

exit()
