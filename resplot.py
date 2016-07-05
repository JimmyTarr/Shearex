#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1]
saveplot = int(sys.argv[2])

A=np.load(fname)
k1res,k2res,kmax,szk,gam1map,gam2map,Kappa,pix=A['k1res'],A['k2res'],A['kmax'],A['szk'],A['gam1map'],A['gam2map'],A['Kappa'],A['pix']
A.close()

#####################Create Axis & grids
kax=np.linspace(-kmax,kmax,szk)
k1,k2=np.meshgrid(kax,kax)
szx=gam1map.shape[0]
u=np.fft.fftshift(np.fft.fftfreq(szx,pix))
uu,vv=np.meshgrid(u,u,sparse=1)

print "Starting Analysis"

#####################Generate True Fkappa and FGamma

if np.sum(Kappa)<3:
    fgam1in=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gam1map)))
    fgam2in=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gam2map)))
else:
    fKappa=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(Kappa)))
    divu=uu**2+vv**2
    divu[np.where(divu==0)]=1.
    fgam1in=fKappa*(uu**2-vv**2)/divu
    fgam2in=2*fKappa*uu*vv/divu
##
##
###################Generate Kappa Estimates
##        
##divk=k1**2+k2**2
##divk[np.where(divk==0)]=1.
##Fkappa=((k1**2-k2**2)*k1res+2*k1*k2*k2res)/divk
##kappaout=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Fkappa),s=[szx,szx]))
##kappaout=smooth.gaussian_filter(kappaout.real,120,mode='constant')
##kappaout*=np.sum(Kappa)/np.sum(np.abs(kappaout))
##
##
###################Regrid
##
region=np.where(np.abs(u)<=kmax)
a,b=min(region[0]),max(region[0])
fg1i=fgam1in[a:b+1,a:b+1]

print "Ploting"
##
##KDiff=Kappa-kappaout
##Gdiff=((fgam1in-gnorm*fgam1o.real)+(fgam2in-gnorm*fgam2o.real))/(2*np.median(np.abs(fgam1o)))

print stats.pearsonr(fg1i.real.ravel(),k1res.real.ravel())
savename='Gam1invsout'
plt.figure(savename)
s=plt.scatter(fg1i.real,k1res.real)
plt.xlabel('True real space $\gamma_1$ field')
plt.ylabel('Reconstructed $\gamma_1$')
##
if saveplot==0:
    plt.show()
else:
    Loc="/users/mctarr/shearex/Plots"
    a=fname.split('/')[1]
    a=a.split('.')[0]
    plt.savefig(os.path.join(Loc,string.join((a,savename),sep='')))
