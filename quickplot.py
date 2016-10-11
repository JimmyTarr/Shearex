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

szx = gam1map.shape[0]
szk = Res.shape[0]

if real==0:
    In = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gam1map)))
    In = In[szx/2-int(szk/2):szx/2+int(szk/2)+1,szx/2-int(szk/2):szx/2+int(szk/2)+1]
else:
    In = gam1map
    
mult = In.real.max()/Res.real.max()
print 'm',mult
print 'mu','sig',(In-mult*Res)[:,int(szk/2)+1:].mean(),(In-mult*Res)[:,int(szk/2)+1:].std()
print 'SN',Res.max()/((Res-(In/mult))[:,int(szk/2)+1:].std())

plt.figure('In')
plt.pcolor(In.real)
plt.colorbar()
plt.figure('Out')
plt.pcolor(Res.real)
plt.colorbar()
plt.figure('diff')
plt.pcolor((In-mult*Res).real)
plt.colorbar()
plt.show()
plt.close()
