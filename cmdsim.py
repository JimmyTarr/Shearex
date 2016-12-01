#!/usr/bin/python

import shearex, sys
#import numpy as np

skysz,eint,pix=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
Noise=int(sys.argv[5])


if str.isdigit(sys.argv[4][-1]):
    kappa = eval(sys.argv[4])
    #kappa=eval('[' + sys.argv[4] + ']')
else:
    kappa=sys.argv[4]

print skysz,eint,pix,kappa

shearex.simsky(skysz,eint,pix,kappa,Noise)

exit()
