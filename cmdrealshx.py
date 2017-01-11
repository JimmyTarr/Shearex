#!/usr/bin/python

import shearex, sys
import numpy as np

print sys.argv

a=float(sys.argv[2])

shearex.shxreal(sys.argv[1],a)

print "Job Complete"

exit()
