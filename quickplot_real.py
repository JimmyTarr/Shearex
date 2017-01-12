#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter as smooth
import scipy.ndimage.interpolation as scale
import matplotlib.pyplot as plt
import matplotlib.colors as cls

fname=sys.argv[1]
if sys.argv[2]=='2':
    go = 'gam2res'
    gi = 'gam2map'
else:
    go = 'gam1res'
    gi = 'gam1map'

A=np.load(fname)
gamt,gamr,pix,posmax,szg,ngal = A[gi],A[go].real,A['pix'],A['posmax'],A['szg'],A['ngal']
b1,b2 = A['b1'],A['b2']
A.close()

szx = gamt.shape[0]
xs,gpix = np.linspace(-posmax,posmax,szg,retstep=1)

gamts = np.zeros((szg,szg))
nadv = szx/szg
for i in range(szg):
    for j in range(szg):
        gamts[i,j] = gamt[nadv*i:nadv*(i+1),nadv*j:nadv*(j+1)].mean()

smth = 0
if smth>0:
#    gamt = smooth(gamt,nadv*smth)
    gamts = smooth(gamts,smth)
    gamr = smooth(gamr,smth)

##m,c,r,p,stderr = stats.linregress(gamts.ravel(),gamr.ravel())
##gamout = (gamout-c)/m
##diff = gamin-gamout
##print 'mc',m,c
##print 'rp',r,p
##print 'SN',np.abs(gamin).mean()/diff.std()

##plt.figure('In')
###plt.imshow(gamt,origin='lower',extent=(-600,600,-600,600),interpolation='nearest')
plt.figure('In small')
plt.pcolor(xs,xs,gamts)
plt.figure('out')
plt.pcolor(xs,xs,gamr)
plt.figure('in vs out')
plt.plot(gamts.ravel(),gamr.ravel(),'.')
plt.show()




