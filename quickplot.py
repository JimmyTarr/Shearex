#!/usr/bin/python

import sys
import numpy as np
##import scipy.stats as stats
##import scipy.ndimage.interpolation as scale
import matplotlib.pyplot as plt

fname=sys.argv[1]
if str.isdigit(fname[8]):
    print 'Fourier'
    real=0
else:
    print 'Real'
    real=1

A=np.load(fname)
Res,gam1map,pix,kmax=A['k1res'],A['gam1map'],A['pix'],A['kmax']
A.close()

if real==0:
    fgam = np.fft.fftshift(np.fft.fftn(gam1map))

mult = fgam.real.max()/Res.real.max()

plt.figure('In')
plt.pcolor(fgam.real)
plt.colorbar()
plt.figure('Out')
plt.pcolor(Res.real)
plt.colorbar()
plt.figure('diff')
plt.pcolor((fgam-mult*Res).real)
plt.colorbar()
plt.show()
plt.close()
