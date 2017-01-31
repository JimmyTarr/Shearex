#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
##from scipy.ndimage.filters import gaussian_filter as smooth
##import scipy.ndimage.interpolation as scale
import matplotlib.pyplot as plt
##import matplotlib.colors as cls

fname=sys.argv[1]
smth = 1

A=np.load(fname)
gam1t,gam2t,fgam1r,fgam2r=A['gam1map'],A['gam2map'],A['k1res'],A['k2res']
pix = A['pix']
A.close()

szx = gam1t.shape[0]
szk = fgam1r.shape[0]
nbins = 15

if smth:
    u = np.fft.fftshift(np.fft.fftfreq(szk,pix*szx/szk))
    uu,vv = np.meshgrid(u,u,sparse=1)
    r = np.sqrt(uu**2+vv**2)
    fgam1r[r>0.008]=0
    fgam2r[r>0.008]=0

gam1r = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fgam1r))).real
gam2r = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fgam2r))).real
gam1ts = gam1t.reshape(szk,szx/szk,szk,-1).mean((1,3))
gam2ts = gam2t.reshape(szk,szx/szk,szk,-1).mean((1,3))
    
gamr = gam1r+complex(0,1)*gam2r
gamts = gam1ts+complex(0,1)*gam2ts
print 'r',stats.linregress(gamr.ravel(),gamts.ravel())[2]
