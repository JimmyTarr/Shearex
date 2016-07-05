#!/usr/bin/python

import sys, string, os
import numpy as np

partname=sys.argv[1]

if partname.split('-')[0]=='real':
    print "real"
    fname="0-",partname,
    fname=string.join(fname,sep="")
    wfile=open(os.path.join("temp",fname),'r')
    A=np.load(wfile)
    szk=A['szk']
    wfile.close()
    k1res,k2res,ngal=0,0,0
    for i in range(int(szk)):
        fname=str(int(i)),"-",partname,
        fname=string.join(fname,sep="")
        wfile=open(os.path.join("temp",fname),'r')
        A=np.load(wfile)
        t1,t2,t3=A['gam1res'],A['gam2res'],A['ngal']
        k1res+=t1
        k2res+=t2
        ngal+=t3
        wfile.close()
    wfile=open(os.path.join("temp",fname),'r')
    A=np.load(wfile)
    kmax,gam1map,gam2map,Kappa,pix,notes=A['posmax'],A['gam1map'],A['gam2map'],A['Kappa'],A['pix'],str(A['notes'])
    wfile.close()

    ngal[ngal==0] = 1.
    k1res[ngal==0] = 0
    k2res[ngal==0] = 0
    k1res*=1./ngal
    k2res*=1./ngal

else:
    print 'Fourier'
    fname="0-",partname,
    fname=string.join(fname,sep="")
    wfile=open(os.path.join("temp",fname),'r')
    A=np.load(wfile)
    szk,rows=A['szk'],A['rows']
    wfile.close()
    k1res=0
    k2res=0

    print rows
    
    if rows==1:
        n=szk
    elif rows==0:
        n=szk*np.ceil(szk/2)
    for i in range(int(n)):
        fname=str(i),"-",partname
        fname=string.join(fname,sep="")
        fullname=os.path.join("temp",fname)
        wfile=open(fullname,'r')
        A=np.load(wfile)
        t1,t2=A['k1res'],A['k2res']
        k1res+=t1
        k2res+=t2
        wfile.close()

    wfile=open(os.path.join("temp",fname),'r')
    A=np.load(wfile)
    kmax,szk,gam1map,gam2map,Kappa,pix,notes=A['kmax'],A['szk'],A['gam1map'],A['gam2map'],A['Kappa'],A['pix'],A['notes']
    wfile.close()

k1res=np.transpose(k1res)
k2res=np.transpose(k2res)

wfile=open(os.path.join("Results",partname),'w')
np.savez(wfile,k1res=k1res,k2res=k2res,kmax=kmax,szk=szk,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,pix=pix,notes=notes)
wfile.close()
print "Result obtained"

exit()
