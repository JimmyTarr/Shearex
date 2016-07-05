#!/usr/bin/python

import shearex, sys

print sys.argv

#szk=int(sys.argv[3])
kmax=float(sys.argv[2])

shearex.shx(sys.argv[1],kmax)

print "Job Complete"

exit()
