#!/usr/bin/python

import shearex, sys
import numpy as np

skysz,eint,pix=np.float(sys.argv[1]),np.float(sys.argv[2]),np.float(sys.argv[3])
uvfrac=int(sys.argv[5])

if str.isdigit(sys.argv[4]):
    kappa=int(sys.argv[4])
elif str.isdigit(str.split(str.split(sys.argv[4],',')[0],'.')[0]):
    kappa=eval('[' + sys.argv[4] + ']')
else:
    kappa=sys.argv[4]

print skysz,eint,pix,kappa

shearex.simsky(skysz,eint,pix,kappa,uvfrac)

exit()
