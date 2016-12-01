#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
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

A=np.load(fname)
out,gamap,pix,kmax,u=A[gout],A[gin],A['pix'],A['kmax'],A['u']
A.close()
print 'loaded'
szx = gamap.shape[0]
szk = out.shape[0]

u = u[szx/2-szk/2:szx/2+szk/2+1]
uu,vv = np.meshgrid(u,u,sparse=1)
umax = np.abs(u.max()+0.9*np.gradient(u)[0])
Res = out*(uu**2+vv**2<umax**2)
#Res *= np.sinc((uu**2+vv**2)**0.5/umax)

In = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gamap)))
In = In[szx/2-int(szk/2):szx/2+int(szk/2)+1,szx/2-szk/2:szx/2+int(szk/2)+1]

m,c,r,p,stderr = stats.linregress(In[uu**2+vv**2<umax**2],Res[uu**2+vv**2<umax**2])
Res[uu**2+vv**2<umax**2] -= c
Res *= 1./m.real
diff = In-Res
print 'mc',m,c
print 'rps',r,p,stderr
print 'SN',In.std()/diff.std()

n=10
tempt = In[uu**2+vv**2<umax**2].ravel()
tempr = Res[uu**2+vv**2<umax**2].ravel()
tempd = diff[uu**2+vv**2<umax**2].ravel()
htr,bt = np.histogram(tempt.real,n)
hti = np.histogram(tempt.imag,bt)[0]
n *= np.ceil((tempd.real.max()-tempd.real.min())/(bt.max()-bt.min()))
hdr,bd = np.histogram(tempd.real,n)
hdi = np.histogram(tempd.ravel(),bd)[0]

##plt.figure()
##plt.pcolor((diff).real)
##plt.colorbar()
plt.figure('In vs Out')
plt.scatter(tempt.real,tempr.real,15,'g')
plt.scatter(tempt.imag,tempr.imag,15,'b')
plt.plot(np.linspace(-In.real.max(),In.real.max(),10),np.linspace(-In.real.max(),In.real.max(),10),'r')
plt.figure('Signal and Noise')
plt.plot(bt[:-1],htr,'g')
plt.plot(bt[:-1],hti,'g--')
plt.plot(bd[:-1],hdr,'r')
plt.plot(bd[:-1],hdi,'r--')
plt.show()
plt.close()
