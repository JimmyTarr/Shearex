#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits

na = 250
nb = na*(na-1)/2
UKM = np.random.normal(0,50,nb)
VKM = np.random.normal(0,50,nb)
sel = UKM>=0
UKM = np.append(UKM[sel],-1*UKM[sel])
VKM = np.append(VKM[sel],-1*VKM[sel])
badvals = (UKM**2+VKM**2>150**2)
UKM = UKM[badvals==0]
VKM = VKM[badvals==0]
WKM = np.zeros(nb,dtype=int)

tbhdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name='U',format='F',array=UKM),
        pyfits.Column(name='V',format='F',array=VKM),
        pyfits.Column(name='W',format='F',array=WKM)])
tbhdu.writeto('uvtest'+str(na)+'.fits')

exit()
