#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter as smooth
#import scipy.ndimage.interpolation as scale
import matplotlib.pyplot as plt
import matplotlib.colors as cls

fname=sys.argv[1]
if sys.argv[2]=='2':
    gout = 'k2res'
    gin = 'gam2map'
else:
    gout = 'k1res'
    gin = 'gam1map'

A=np.load(fname)
gamout,gami,pix = A[gout].real,A[gin],A['pix']
A.close()

szx = gami.shape[0]
szk = gamout.shape[0]

gamin = np.zeros((szk,szk))
nadv = szx/szk
for i in range(szk):
    for j in range(szk):
        gamin[i,j] = gami[nadv*i:nadv*(i+1),nadv*j:nadv*(j+1)].mean()

smth = float(sys.argv[3])
#if smth>0:
    gamin = smooth(gamin,1)#,smth)
    gamout = smooth(gamout,1)#,smth)

m,c,r,p,stderr = stats.linregress(gamin.ravel(),gamout.ravel())
gamout = (gamout-c)/m
diff = gamin-gamout
print 'mc',m,c
print 'rp',r,p
print 'SN',np.abs(gamin).mean()/diff.std()

n=20
ht,bt = np.histogram(gamin.ravel(),n)
hd,bd = np.histogram(gamout.ravel(),n)
plt.rcParams['image.cmap'] = 'bwr'
plt.figure('In')
plt.pcolor(gamin,norm=cls.Normalize(-np.abs(gamin).max(),np.abs(gamin).max()))
plt.colorbar()
plt.figure('Out')
plt.pcolor(gamout,norm=cls.Normalize(-np.abs(gamout).max(),np.abs(gamout).max()))
plt.colorbar()
plt.figure('In vs Out')
plt.scatter(gamin.ravel(),gamout.ravel(),15,'g')
plt.plot(np.linspace(-gamin.max(),gamin.max(),10),np.linspace(-gamin.max(),gamin.max(),10),'r')
plt.figure('Signal and Noise')
plt.plot(bt[:-1],ht,'g')
plt.plot(bd[:-1],hd,'r')
plt.show()
plt.close()
