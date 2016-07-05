#!/usr/bin/python

import shearex, sys
import numpy as np

print sys.argv

a=float(sys.argv[3])
b=int(sys.argv[4])-1

shearex.shxreal(sys.argv[2],b,sys.argv[1],a)

print "Job Complete"

exit()
