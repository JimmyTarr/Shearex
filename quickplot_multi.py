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

In = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gamap)))
In = In[szx/2-int(szk/2):szx/2+int(szk/2)+1,szx/2-szk/2:szx/2+int(szk/2)+1]

m,c,r,p,stderr = stats.linregress(In.ravel(),Res.ravel())
In *= m
Res -= c
diff = In-Res
print 'mc',m,c
print 'rps',r,p
print 'SN',In.std()/diff.std()

n=15
htr,bt = np.histogram(In.real.ravel(),n)
hti = np.histogram(In.imag.ravel(),bt)[0]
hdr,bd = np.histogram(diff.real.ravel(),n)
hdi = np.histogram(diff.imag.ravel(),bd)[0]

##n-=10
##sigdiff = np.zeros(n)
##bk = np.linspace(0,np.abs(u).max()**2,n+1)
##for i in range(n):
##    sigdiff[i] = diff[((uu**2+vv**2>=bk[i])*(uu**2+vv**2<bk[i+1]))].std()
##plt.figure('In vs diff spread')
##plt.plot(bk[:-1]**0.5,sigdiff)
plt.figure('In vs Out')
plt.scatter(In.real.ravel(),Res.real.ravel(),15,'g')
plt.scatter(In.imag.ravel(),Res.imag.ravel(),15,'b')
plt.plot(np.linspace(-In.real.max(),In.real.max(),10),np.linspace(-In.real.max(),In.real.max(),10),'r')
plt.figure('Signal and Noise')
plt.plot(bt[:-1],htr,'g')
plt.plot(bt[:-1],hti,'g--')
plt.plot(bd[:-1],hdr,'r')
plt.plot(bd[:-1],hdi,'r--')
plt.show()
plt.close()
