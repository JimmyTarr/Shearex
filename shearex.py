import math, os, string, sys, time
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intp
import scipy.ndimage.interpolation as scale
import scipy.stats as stats

def simsky(skysz=2,eint=0.,pix=0.2,Kappa=(30,0.1),uvfrac=0):
    stime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]
    ###############################Inputs#########################################

    #Notes
    notes = 'skysz='+str(skysz)+' eint='+str(eint)+' pix='+str(pix)
    
    #Filename Extras
    Extra='grid'
    
    #UV Frac
    if uvfrac>0:
        Extra='UVfrac',str(int(uvfrac*10))
        Extra=string.join(Extra,sep='')

    #Galaxy population parameters
    #Profile: Sirsic index, 0.5=Gaussian, 1=Exponential disk...
    fluxlim = 5*10**(-6)
    notes+=' fluxlimit='+'%g'%fluxlim
    profile = 0.5
    KappaZ = 0. #Redshift of Kappa
    fgrds = 0 #Include Forgournd galaxies
    lobes = 0 #Include Lobe sources
    points = 0 #Include point sources (Cores and Hotspots)

    #Load Wilman
    hdu = pyfits.open('Wilman/SEX.fits')
    catcomp = hdu[1].data['structure']
    catra = hdu[1].data['right_ascension']
    catra*=60*60#change units to arcsecs
    catdec = hdu[1].data['declination']
    catdec*=60*60#change units to arcsecs
    catsize = hdu[1].data['size']
    catflux = np.power(10,hdu[1].data['i_151'])
    catz = hdu[1].data['redshift']
    hdu.close()
    catn = len(catra)

    ###############################Image set-up###########################

    rng = math.ceil(skysz*60./2.)
    catracen = np.random.uniform(catra.min()+rng,catra.max()-rng)
    catdeccen = np.random.uniform(catdec.min()+rng,catdec.max()-rng)
    notes+=' cen='+str(catracen)+','+str(catdeccen)

    pad=5*np.sort(catsize)[np.ceil(len(catsize)*0.9)]
    rngp = rng-pad
    selectidx = (catra>catracen-rngp)*(catra<catracen+rngp)*(catdec>catdeccen-rngp)*(catdec<catdeccen+rngp)

    print "Creating Array"
    
    x=np.arange(-rng,rng,pix)
    x[np.abs(x)<0.5*pix] = 0
    xx,yy=np.meshgrid(x,x)
    szx=np.size(x)

    ###############################Select Galaxies#########################

    selectidx*=(catflux>fluxlim)
    if fgrds==0:
        selectidx*=(catz>KappaZ)
        notes+=' noforegrounds'
    if lobes==0:
        selectidx*=(catcomp!=2)
        notes+=' no lobes'
    if points==0:
        selectidx*=(catcomp!=1)*(catcomp!=3)
        notes+=' no point sources'

    b1 = catra[selectidx]-catracen
    b2 = catdec[selectidx]-catdeccen
    mag = catflux[selectidx]
    sig = catsize[selectidx]
    z = catz[selectidx]
    component = catcomp[selectidx]
    lensed = (z>KappaZ)*((component==2)+(component==4))
    s = len(b1)#number of galaxy components
    notes+='_n lensed sources='+str(lensed.sum())

    ####For Gridded simple Gals
    Ggals=1
    if Ggals==1:
        s = np.ceil((s**0.5))**2
        bs = np.floor(s**0.5)
        b1 = rngp-2*rngp*(np.arange(s)%bs)/(bs-1)
        b2 = rngp-2*rngp*np.floor(np.arange(s)/bs)/(bs-1)
        mag = np.ones(s)
        sig = 2*np.ones(s)
        z = np.ones(s)
        component = 4*np.ones(s)


    #############################Lensing signal########################
    
    if type(Kappa) is str:
        Ktype = 3
        GammaMax=0.
        hdu=pyfits.open('Kappainputs/CFHTLenS-w1.fits')
        Kappafull = hdu[0].data
        hdu.close()
    elif len(Kappa)==2:
        Ktype = 2
        kmin,SigGamma = Kappa[0],Kappa[1]*2**(-0.5)
    elif len(Kappa)==5:
        Ktype = 1
        amp,x1,y1,x2,y2 = Kappa
        GammaMax = amp
    else:
        print 'Unknown kappa type'
        

    ###############################Lensing Maps###########################

    print "Generating Gamma Maps"

    if Ktype<3:
        u = np.fft.fftfreq(szx,pix)
        if Ktype==2:
            umax = np.abs(np.fft.fftfreq(szx,kmin)).max()
            ksz = (np.abs(u)<=umax+0.9*u[1]).sum()
            fgam = ksz*np.random.normal(0,SigGamma,(4,ksz,int(ksz/2.)+1))
            fgam1 = np.zeros((szx,int(szx/2)+1))
            fgam2 = np.zeros((szx,int(szx/2)+1))
            srtidx = int(szx/2)-int((szk-1)/2)
            fgam1[srtinx:srtind+ksz,0:int(ksz/2.)+1] += fgam[0,:,:]+complex(0,1)*fgam[1,:,:]
            fgam2[srtinx:srtind+ksz,0:int(ksz/2.)+1] += fgam[2,:,:]+complex(0,1)*fgam[3,:,:]
            fgam1 = np.fft.fftshift(fgam1,0)
            fgam2 = np.fft.fftshift(fgam2,0)
        else:
            x1coord = np.round(x1/u[1])
            y1coord = np.abs(np.round(y1/u[1]))
            x2coord = np.round(x2/u[1])
            y2coord = np.abs(np.round(y2/u[1]))
            fgam1 = np.zeros((szx,int(szx/2)+1))
            fgam2 = np.zeros((szx,int(szx/2)+1))
            fgam1[x1coord,y1coord] = szx**2*amp
            fgam2[x2coord,y2coord] = szx**2*amp
        gam1map = np.fft.irfftn(fgam1,(szx,szx))
        gam2map = np.fft.irfftn(fgam2,(szx,szx))
    else:
        kxstart = np.random.randint(Kappafull.shape[1]-skysz-1)
        kystart = np.random.randint(Kappafull.shape[0]-skysz-1)
        Kappa = Kappafull[kystart:kystart+skysz,kxstart:kxstart+skysz]
        Kappa = scale.zoom(Kappa,float(szx)/skysz)
        fKappa=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(Kappa)))
        u=np.fft.fftshift(np.fft.fftfreq(szx,pix))
        uu,vv=np.meshgrid(u,u,sparse=1)
        div=uu**2+vv**2
        div[np.where(div==0)]=1.
        fgam1=fKappa*(uu**2-vv**2)/div
        fgam2=2*fKappa*uu*vv/div
        div=None
        gam1map=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fgam1))).real
        gam2map=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fgam2))).real
        uu,vv=2*[None]
        mult=GammaMax/np.max((np.abs(gam1map),np.abs(gam2map)))
        gam1map*=mult
        gam2map*=mult
    
    
    xx,yy=2*[None]
    print "Extracting Gamma Apmlitudes"

    gamfac=np.power(2,-0.5)#factor of root two scales e for gamma
    if eint==0:
        gam1 = np.zeros(s)
        gam2 = np.zeros(s)
    else:
        gam1 = np.random.normal(0,eint*gamfac,s)
        gam2 = np.random.normal(0,eint*gamfac,s)
    for i in range(int(s)):
        invpix = (len(x)-1)/(x.max()-x.min())
        xcoord = np.round((b1[i]-x[0])*invpix)
        ycoord = np.round((b2[i]-x[0])*invpix)
        gam1[i]+= gam1map[ycoord,xcoord]
        gam2[i]+= gam2map[ycoord,xcoord]
    
    ################################Data Simulation###############################

    ######Image Domain
    print  "Building Sky"

    f=np.zeros((szx,szx))
    for i in range(int(s)):
        if (component[i]==1)+(component[i]==3):
            indx = np.where(np.abs(x-b1[i]).min()==np.abs(x-b1[i]))
            indy = np.where(np.abs(x-b2[i]).min()==np.abs(x-b2[i]))
            f[indy,indx]+=mag[i]
        else:
            cutrng=10*sig[i]
            indx=np.where((x>=b1[i]-cutrng)&(x<=b1[i]+cutrng))
            indy=np.where((x>=b2[i]-cutrng)&(x<=b2[i]+cutrng))
            xtemp=x[indx]-b1[i]
            ytemp=x[indy]-b2[i]
            xb,yb=np.meshgrid(xtemp,ytemp,sparse=1)
            if z[i]<KappaZ:
                gam1t,gam2t=0,0
            else:
                gam1t=gam1[i]
                gam2t=gam2[i]
            temp=(xb-gam1t*xb-gam2t*yb)**2.+(yb+gam1t*yb-gam2t*xb)**2.
            #temp2=(1/sig[i])*(1.-gam1t**2.-gam2t**2.)**-2
            ftemp=mag[i]*np.exp(-(temp/sig[i])**(1/(2*profile)))
            a,b,c,d=indx[0][0],indx[0][-1],indy[0][0],indy[0][-1]
            f[c:d+1,a:b+1]+=ftemp
##        print "source",i,"complete"
    xb,yb,temp1,temp2,ftemp=5*[None]

    ######Fourier Domain

    print "Simulating Radio obs"
    
    ##Numerical    
    num=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
    szu=np.shape(num)[0]
    u=np.fft.fftshift(np.fft.fftfreq(szu,pix))

    ##Calculate Antenna noise\ Sampling pattern
##    rms = 0.2*fluxlim
##    frms = rms*np.sqrt(num.size/2)
##    fnoise = np.random.normal(0,frms,(szu,szu))+complex(0,1)*np.random.normal(0,frms,(szu,szu))
##    num += fnoise
    
    if uvfrac>0:
        uu,vv=np.meshgrid(u,u,sparse=1)
        dist=np.sqrt(uu**2+vv**2)
        mask=np.zeros((szu,szu))
        if uvfrac==1:
            div=1
        else:
            div=uvfrac-1
        starts=0.9*np.max(dist)*np.arange(int(uvfrac))/div
        for i in starts:
            idx=np.where((dist>i)&(dist<=i+0.1*np.max(dist)))
            mask[idx]=1
        num*=mask
    
    #############################Write data#######################

    print "Saving"

    Path="Sims"
    fname=str(int(skysz)),"_",str(int(60/pix)),"_",str(int(10*eint)),"_",str(int(Ktype)),"_",Extra,".dat"
    fname=string.join(fname,sep="")
    wfile=open(os.path.join(Path,fname),'wb')
    np.savez(wfile,sig=sig,mag=mag,b1=b1,b2=b2,u=u,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,num=num,pix=pix,profile=profile,lensed=lensed,notes=notes)
    wfile.close()
    notes+=' '+fname

    runtime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]-stime
    H=divmod(runtime,3600)[0]
    M=divmod(divmod(runtime,3600)[1],60)[0]
    S=divmod(divmod(runtime,3600)[1],60)[1]
    print "Run time =",H,M,S
    print "Gamma Max =",np.max((gam1,gam2))
    print notes

##    #############################Tests###########################
##
##    ###########################Analytical
##    uu,vv=np.meshgrid(u,u)
##    mfactor=(2*min(u))**2
##
##    xx=None
##
##    E=mbar*np.pi*sbar*np.exp(-np.pi**2.*sbar*(uu**2+vv**2))
##    uk1=2*np.pi**2*sbar*(uu**2-vv**2)
##    uk2=4*np.pi**2*sbar*uu*vv
##
##    ana1=np.zeros_like(num)
##    for i in range(int(s)):
##        P=np.exp(-2.*math.pi*complex(0,1)*(b1[i]*uu+b2[i]*vv))
##        ana1+=P*(1-uk1*gam1[i]-uk2*gam2[i])
##    ana1*=E*mfactor
##
##    uu,vv=None,None
##
##    ##########################Sky check
####    
##    skyback=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ana1))).real
##    sizefac=1
##    print 'plot'
##
##    plt.figure('Residules')
##    a=scale.zoom((num-ana1).real,sizefac)
##    approxsky=plt.pcolor(a)
##    plt.colorbar(approxsky)
##
##    plt.figure('Gamma1map')
##    approxsky=plt.pcolor(scale.zoom(gam2map.real,sizefac))
##    plt.colorbar(approxsky)
##
##    plt.figure('truesky')
##    truesky=plt.pcolor(scale.zoom(f,sizefac))
##    plt.colorbar(truesky)
##
##    plt.figure('SkyCheck')
##    res=plt.pcolor(scale.zoom(f-skyback,sizefac))
##    plt.colorbar(res)
##

##    plt.show()
##    plt.savefig('/users/mctarr/shearex/Plots/')

##########################################################################################################################################################################################################################################################################################################################################################################################

def shx(fname,kmax=0.008):
    stime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]    
    
##    #############################Import data#######################

    A=np.load(os.path.join("Sims",fname))
    sig,mag,b1,b2,u,gam1map,gam2map,Kappa,num,pix,profile,lensed,notes=A['sig'],A['mag'],A['b1'],A['b2'],A['u'],A['gam1map'],A['gam2map'],A['Kappa'],A['num'],A['pix'],A['profile'],A['lensed'],str(A['notes'])
    A.close()

    b1=b1[lensed]
    b2=b2[lensed]

##    ###########################Create???##########################
        
    szu,s=u.size,b1.size
    uu,vv=np.meshgrid(u,u,sparse=1)
    upix=np.gradient(u)[0]
    
    #kax = u[np.abs(u)-kmax<=0.5*np.gradient(u)[0]]#np.linspace(-kmax,kmax,szk)
    #szk = kax.size
    szk = len(gam1map)
    kax = np.fft.fftshift(np.fft.fftfreq(szk,60))
    kx,ky = np.meshgrid(kax,kax)
    k1res=np.zeros([szk,szk],dtype=complex)
    k2res=np.zeros([szk,szk],dtype=complex)

##    #########################Do#################################

    uk1=(uu**2-vv**2)
    uk2=2*uu*vv

    w = np.ones(s)
##    denr = 50
##    nexp = np.pi*s*denr**2/((szu*pix)**2)
##    for i in range(s):
##        w[i] = nexp/np.sum(np.sqrt((b1-b1[i])**2+(b2-b2[i])**2)<denr)
        
    eu = np.exp(2*np.pi*uu)
    ev = np.exp(2*np.pi*vv)
    ekx = np.exp(2*np.pi*kx)
    eky = np.exp(2*np.pi*ky)
    for i in range(s):
        VP = num*np.power(eu,complex(0,b1[i]))*np.power(ev,complex(0,b2[i]))
        fg1 = np.sum(uk1*VP)
        fg2 = np.sum(uk2*VP)
        ekb = np.power(ekx,complex(0,b1[i]))*np.power(eky,complex(0,b2[i]))
        k1res += w[i]*fg1*ekb
        k2res += w[i]*fg2*ekb
    mult = -1#upix**2
    k1res *= mult
    k2res *= mult
    
################### Write ###################################

    wfile=open(os.path.join("Results",fname),'wb')
    np.savez(wfile,k1res=k1res,k2res=k2res,kmax=kmax,szk=szk,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,pix=pix,notes=notes)
    wfile.close()

    runtime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]-stime
    H=divmod(runtime,3600)[0]
    M=divmod(divmod(runtime,3600)[1],60)[0]
    S=divmod(divmod(runtime,3600)[1],60)[1]
    print "Run time =",H,M,S

###################################################################################
#####################################################################################

def shxreal(fname,istart,Loc="/users/mctarr/shearex",gpix=60):
    stime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]
    
##    #############################Import data#######################
    
    wfile=open(os.path.join(Loc,'Sims',fname),'r')
    A=np.load(wfile)
    sig,mag,b1,b2,u,gam1map,gam2map,Kappa,num,pix,lensed,notes=A['sig'],A['mag'],A['b1'],A['b2'],A['u'],A['gam1map'],A['gam2map'],A['Kappa'],A['num'],A['pix'],A['lensed'],str(A['notes'])
    wfile.close()

    b1=b1[lensed]
    b2=b2[lensed]

##    ###########################Create???##########################

    posmax=u.size*pix
    szg=np.ceil(posmax/gpix)

    iend=istart+1
    if iend > szg:
        iend = int(szg)
##    print iend
        
    szu,s=u.size,b1.size
    uu,vv=np.meshgrid(u,u,sparse=1)
    upix=u[1]-u[0]
    eu=np.exp(-2.*np.pi*uu)
    ev=np.exp(-2.*np.pi*vv)

    posmax=10*np.floor(posmax*(szg-1)/(20*szg))
    gax=np.linspace(-posmax,posmax,szg,retstep=1)
    gpix=gax[1]
    gax=gax[0]
    gam1res=np.zeros([szg,szg],dtype=complex)
    gam2res=np.zeros([szg,szg],dtype=complex)
    ngal=np.zeros([szg,szg])

    
##    #########################Do#################################

    uk1=2*np.pi**2*sig.mean()*(uu**2-vv**2)
    uk2=4*np.pi**2*sig.mean()*uu*vv

##    print 'starting analysis'
    pg=0
    for i in range(istart,iend):
        xcen=gax[i]
        for j in range(int(szg)):
            ycen=gax[j]
            gsub=np.where((np.abs(b1-xcen)<gpix/2.) & (np.abs(b2-ycen)<gpix/2.))
            for g in gsub[0]:
                pg+=np.power(eu,complex(0,1)*b1[g])*np.power(ev,complex(0,1)*b2[g])
            gres=num*np.conj(pg)
            gam1res[i,j]=-upix*upix*np.sum(uk1*gres)
            gam2res[i,j]=-upix*upix*np.sum(uk2*gres)
            ngal[i,j]=gsub[0].size
            pg=0
    pg,num,uk1,eu,ev=5*[None]
    

################### Write ###################################


    filename=str(int(istart)),"-real-",fname
    filename=string.join(filename,sep="")
    wfile=open(os.path.join(Loc,"temp",filename),'wb')
    np.savez(wfile,gam1res=gam1res,gam2res=gam2res,ngal=ngal,posmax=posmax,szk=szg,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,pix=pix,notes=notes)
    wfile.close()

    runtime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]-stime
    H=divmod(runtime,3600)[0]
    M=divmod(divmod(runtime,3600)[1],60)[0]
    S=divmod(divmod(runtime,3600)[1],60)[1]
    print "Run time =",H,M,S

##########################################################################
##Old scripts
#####################################################################

def preproc(fname):

    stime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]
    Loc='/users/mctarr/shearex'
    
    wfile=open(os.path.join(Loc,"Sims",fname),'r')
    A=np.load(wfile)
    sig,mag,b1,b2,u,gam1map,gam2map,Kappa,num,pix,profile=A['sig'],A['mag'],A['b1'],A['b2'],A['u'],A['gam1map'],A['gam2map'],A['Kappa'],A['num'],A['pix'],A['profile']
    wfile.close()

    szu,s=u.size,b1.size
    uu,vv=np.meshgrid(u,u,sparse=1)
    upix=u[1]-u[0]

    
    rng=szu*pix*0.5
    x=np.arange(-rng,rng,pix)
    cutrng=np.ceil(15.*sig.mean())
    f=np.zeros((szu,szu))
    propow=1/(2.*profile)
    const=-sig**(-propow)
    for i in range(int(s)):
        indx=np.where((x>=b1[i]-cutrng)&(x<=b1[i]+cutrng))
        indy=np.where((x>=b2[i]-cutrng)&(x<=b2[i]+cutrng))
        xtemp=(x[indx]-b1[i])**2.
        ytemp=(x[indy]-b2[i])**2.
        xb2,yb2=np.meshgrid(xtemp,ytemp,sparse=1)
        ftemp=mag[i]*np.exp(-((xb2+yb2)/sig[i])**propow)
        a,b,c,d=indx[0][0],indx[0][-1],indy[0][0],indy[0][-1]
        f[c:d+1,a:b+1]+=ftemp
    f=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f)))
    x,xb2,yb2=3*[None]

    idx=np.where(np.abs(num)<10**(-5))
    if idx[0].size>0:
        f[idx]=0
    num-=f
    
    
    wfile=open(os.path.join(Loc,"PreProc",fname),'w')
    np.savez(wfile,sig=sig,mag=mag,b1=b1,b2=b2,u=u,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,num=num,pix=pix,profile=profile)
    wfile.close()

    runtime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]-stime
    H=divmod(runtime,3600)[0]
    M=divmod(divmod(runtime,3600)[1],60)[0]
    S=divmod(divmod(runtime,3600)[1],60)[1]
    print "Run time =",H,M,S

#########################################################################
########################################################################

def shxold(fname,start,szk,kmax=0.005,rows=1):
    stime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]
    
###################################Check szk and istart

    if szk%2==0:
        szk+=1
    
    if rows==1:
        if start >= szk:
            print 'ERROR: Job number greater than requested resolution requires.'
            raise SystemExit
        istart=int(start)
        jstart=0
        jend=szk
    elif rows==0:
        if start >= szk**2:
            print 'ERROR: Job number greater than requested resolution requires.'
            raise SystemExit
        istart=int(np.floor(start/szk))
        jstart=int(start%szk)
        jend=jstart+1
        
    
##    #############################Import data#######################

    wfile=open(os.path.join("Sims",fname),'r')
    A=np.load(wfile)
    sig,mag,b1,b2,u,gam1map,gam2map,Kappa,num,pix,profile,lensed,notes=A['sig'],A['mag'],A['b1'],A['b2'],A['u'],A['gam1map'],A['gam2map'],A['Kappa'],A['num'],A['pix'],A['profile'],A['lensed'],str(A['notes'])
    wfile.close()

    b1=b1[lensed]
    b2=b2[lensed]

##    ###########################Create???##########################
        
    szu,s=u.size,b1.size
    uu,vv=np.meshgrid(u,u,sparse=1)
    upix=u[1]-u[0]
    
    kax=np.linspace(-kmax,kmax,szk)
    k1res=np.zeros([szk,szk],dtype=complex)
    k2res=np.zeros([szk,szk],dtype=complex)

##    #########################Do#################################

    uk1=2*np.pi**2*sig.mean()*(uu**2-vv**2)
    uk2=4*np.pi**2*sig.mean()*uu*vv

    pgam=np.zeros((szu,szu),dtype=complex)
    print 'starting analysis'
    for i in range(istart,istart+1):
        euk=np.exp(-2*np.pi*(uu+kax[i]))
        for j in range(jstart,jend):
            evk=np.exp(-2*np.pi*(vv+kax[j]))
            for g in range(int(s)):
                pgam+=np.power(euk,complex(0,1)*b1[g])*np.power(evk,complex(0,1)*b2[g])
            kres=num*np.conj(pgam)
            k1res[i,j]=-(upix**2)*np.sum(uk1*kres)
            k2res[i,j]=-(upix**2)*np.sum(uk2*kres)
            pgam=np.zeros_like(pgam)
    pgam,num,gam1,uk1,euk,evk=6*[None]
    

################### Write ###################################

    wfile=open(os.path.join("temp",str(int(start))+"-old-"+fname),'w')
    np.savez(wfile,k1res=k1res,k2res=k2res,kmax=kmax,szk=szk,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,pix=pix,rows=rows,notes=notes)
    wfile.close()

    runtime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]-stime
    H=divmod(runtime,3600)[0]
    M=divmod(divmod(runtime,3600)[1],60)[0]
    S=divmod(divmod(runtime,3600)[1],60)[1]
    print "Run time =",H,M,S

##################################################################
##################################################################      
        
