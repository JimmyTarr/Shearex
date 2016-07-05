#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
import scipy.ndimage.interpolation as scale
import matplotlib.pyplot as plt

fname=sys.argv[1]
if fname.split('-')[0].split('/')[1]=='real':
    print 'real'
    real=1
else:
    print 'Fourier'
    real=0

A=np.load(fname)
Res,gam1map,pix,kmax=A['k1res'],A['gam1map'],A['pix'],A['kmax']
A.close()

if real==0:
    infull = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(gam1map)))
    u = np.fft.fftshift(np.fft.fftfreq(infull.shape[0],pix))
    idx = np.where(np.abs(u)-kmax<=0.5*np.gradient(u)[0])
    a,b = idx[0][0],idx[0][-1]
    Inlow = infull[a:b+1,a:b+1]
    axlow = u[idx]
else:
    infull = gam1map
    rng=gam1map.shape[0]*pix*0.5
    x = np.arange(-rng,rng,pix)
    idx = np.where((x>-kmax)*(x<kmax))
    a,b = idx[0][0],idx[0][-1]
    gam1map = gam1map[a:b+1,a:b+1]
    factor = Res.shape[0]/float(gam1map.shape[0])
    Inlow = scale.zoom(gam1map,factor)
    axlow = scale.zoom(x,factor)
    
norm = stats.linregress(Inlow.real.ravel(),Res.real.ravel())[0]
#In
plt.figure("Gamma1 true shear field")
if real==0:
    in1=plt.pcolor(axlow,axlow,Inlow.real)
else:
    in1=plt.imshow(infull.real,origin='lower')
plt.colorbar(in1)
plt.xlabel('x')
plt.ylabel('y')    
#Out
plt.figure("Gamma1 Reconstructed shear field")
p1=plt.pcolor(axlow,axlow,Res.real)
plt.colorbar(p1)
plt.xlabel('x')
plt.ylabel('y')
###Diff
##plt.figure('Diff Map')
##Diff = (norm*Inlow.real-Res.real)/(norm*Inlow.real)
##d1=plt.pcolor(Diff)
##plt.colorbar(d1)
#In vs Out
plt.figure('In vs out')
plt.scatter(Inlow.real.ravel(),Res.real.ravel()/norm)
plt.xlabel('True real space $\gamma_1$ field')
plt.ylabel('Reconstructed $\gamma_1$')
print stats.pearsonr(Inlow.real.ravel(),Res.real.ravel())
print 'm=',norm,'c=',stats.linregress(Inlow.real.ravel(),Res.real.ravel())[1]/norm


plt.show()
plt.close()
