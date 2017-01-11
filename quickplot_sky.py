#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt

fname=sys.argv[1]

A=np.load(fname)
num,pix=A['num'],A['pix']
A.close()
print 'loaded'

f = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(num))).real


plt.imshow(np.log10(np.abs(f)),origin='lower',interpolation="nearest")
plt.colorbar()
plt.show()
exit()
