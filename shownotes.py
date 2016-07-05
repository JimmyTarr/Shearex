#!/usr/bin/python

from sys import argv
from numpy import load

fname=argv[1]
print load(open(fname))['notes']

exit()
