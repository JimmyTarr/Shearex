#!/usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
#from scipy.ndimage import interpolation as scale
#from scipy import signal
from scipy.ndimage.filters import gaussian_filter as smooth
import matplotlib.pyplot as plt
import matplotlib.colors as cls

fname=sys.argv[1]
gout = 'k1res'
gin = 'gam1map'
if (len(sys.argv)>2):
    if (sys.argv[2]=='2'):
        gout = 'k2res'
        gin = 'gam2map'
if len(sys.argv)>3:
    sfac = int(sys.argv[3])
else:
    sfac = 0

A=np.load(fname)
fgamr,gamt,pix,kmax,u=A[gout],A[gin],A['pix'],A['kmax'],A['u']
b1,b2 = A['b1'],A['b2']
A.close()
print 'loaded'
szx = gamt.shape[0]
szk = fgamr.shape[0]

fgamt = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gamt)))
fgamt = fgamt[szx/2-szk/2:szx/2+szk/2+1,szx/2-szk/2:szx/2+szk/2+1]

u = u[szx/2-szk/2:szx/2+szk/2+1]
uu,vv = np.meshgrid(u,u,sparse=1)
rho = np.sqrt(uu**2+vv**2)
umax = u.max()
upix = np.gradient(u)[0]

nd,xb,yb = np.histogram2d(b1,b2,bins=np.linspace(-szx*pix/2,szx*pix/2,szk+1))
fov = nd>0

gamr = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fgamr))).real

if (szx/szk)%2==0:
    gamt = np.pad(gamt[:-(szx/(2*szk)),:-(szx/(2*szk))],((szx/(2*szk),0),(szx/(2*szk),0)),'constant')
gamt = gamt.reshape(szk,szx/szk,szk,-1).mean((1,3))
x = np.linspace(-szx*pix/2,szx*pix/2,szk)
xx,yy = np.meshgrid(x,x,sparse=1)

gamr = smooth(gamr,sfac)
gamt = smooth(gamt,sfac)
m,c,r,p,stderr = stats.linregress(gamt[fov],gamr[fov])
gamr -= c
gamr /= m
gamr *= fov
gamt *= fov
diff = gamr-gamt
print 'Real space'
print 'mc',m,c
print 'rps',r,p,stderr
print 'SN',diff.std()/gamt.std()

print '...'

fm,fc,fr,fp,fstderr = stats.linregress(fgamt[rho<=umax],fgamr[rho<=umax])
fgamr -= fc
fgamr /= fm.real
fgamr *= rho<=umax
fdiff = fgamr-fgamt
print 'Fourier space'
print 'mc',fm.real,fc
print 'rps',fr.real,fp,fstderr.real
print 'SN',fdiff.std()/fgamt.std()

n=10
ft = fgamt[uu**2+vv**2<=umax**2].ravel()
fr = fgamr[uu**2+vv**2<=umax**2].ravel()
fd = fdiff[uu**2+vv**2<=umax**2].ravel()
bins = np.linspace(np.min((ft.real,fd.real)),np.max((ft.real,fd.real)),n+1)
htr = np.histogram(ft.real,bins)[0]
hti = np.histogram(ft.imag,bins)[0]
hdr = np.histogram(fd.real,bins)[0]
hdi = np.histogram(fd.imag,bins)[0]


##plt.figure('Noise')
##plt.pcolor(np.abs(diff)/gamt.std())
##plt.colorbar()
plt.figure('In vs Out Real')
plt.scatter(gamt.ravel(),gamr.ravel())
plt.plot(np.linspace(-0.1,0.1,3),np.linspace(-0.1,0.1,3),'r')
plt.figure('In vs Out Fourier')
plt.scatter(ft.real,fr.real,15,'g')
plt.scatter(ft.imag,fr.imag,15,'b')
plt.plot(bins,bins,'r')
##plt.figure('Signal and Noise')
##plt.plot(bins[:-1],htr,'g')
##plt.plot(bins[:-1],hti,'g--')
##plt.plot(bins[:-1],hdr,'r')
##plt.plot(bins[:-1],hdi,'r--')
plt.show()
plt.close()
