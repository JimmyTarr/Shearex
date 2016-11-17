#!/usr/bin/python

import sys
import numpy as np
##import scipy.stats as stats
import scipy.ndimage.interpolation as scale
import matplotlib.pyplot as plt
import matplotlib.colors as cls

fname=sys.argv[1]
if sys.argv[2]=='2':
    gout = 'k2res'
    gin = 'gam2map'
else:
    gout = 'k1res'
    gin = 'gam1map'
if str.isdigit(fname[8]):
    print 'Fourier'
    real=0
else:
    print 'Real'
    real=1

A=np.load(fname)
Res,gamap,pix,kmax=A[gout],A[gin],A['pix'],A['kmax']
A.close()

szx = gamap.shape[0]
szk = Res.shape[0]

if real==0:
    Out = Res
    In = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gamap)))
    In = In[szx/2-int(szk/2):szx/2+int(szk/2)+1,szx/2-int(szk/2):szx/2+int(szk/2)+1]
else:
    Out = Res
    In = scale.zoom(gamap,szk/np.float(szx))

mult = np.abs(Out).max()/np.abs(In).max()
In *= mult
diff = In-Out
print 'm',mult
print 'SN',np.abs(Out).max()/(diff[:,int(szk/2)+1:].std())

plt.rcParams['image.cmap'] = 'bwr'
Imax = np.abs(In).max()
Omax = np.abs(Out).max()
Dmax = np.abs(diff).max()
plt.figure('In',figsize=(16,6))
plt.subplot(121)
plt.pcolor(np.abs(In),norm=cls.Normalize(-Imax,Imax))
plt.colorbar()
plt.subplot(122)
plt.pcolor(np.abs(In)*np.angle(In))
plt.colorbar()
plt.figure('Out',figsize=(16,6))
plt.subplot(121)
plt.pcolor(np.abs(Out),norm=cls.Normalize(-Omax,Omax))
plt.colorbar()
plt.subplot(122)
plt.pcolor(np.abs(Out)*np.angle(Out))
plt.colorbar()
plt.figure('diff')
plt.hist((np.sign(diff.real)*np.abs(diff)).ravel())#,norm=cls.Normalize(-Dmax,Dmax))
plt.show()
plt.close()
