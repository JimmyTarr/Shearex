import math, os, string, sys, time
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#import scipy.ndimage.interpolation as scale
#import scipy.stats as stats
#import scipy.interpolate as intp
#from scipy.ndimage.filters import gaussian_filter as smooth

def simsky(skysz=2,eint=0.,pix=0.3,Kappa=(30,0.1),Noise=0):
    stime=60*60*time.gmtime()[3]+60*time.gmtime()[4]+time.gmtime()[5]
    ###############################Inputs#########################################

    #Plot and quit with no save?
    Plot=0

    #Simulate to first sidelobe?
    slsim = 0
    
    #Notes
    notes = 'skysz='+str(skysz)+' eint='+str(eint)+' pix='+str(pix)+'Kappa='+str(Kappa)
    
    #Filename Extras
    Extra=''
    
    #UV mask and Noise
    if Noise==1:
        Extra+='_UVmask'
    elif Noise==2:
        Extra+='_N'
    elif Noise==3:
        Extra+='_N_UVmask'
        
    #Galaxy population parameters
    #Profile: Sirsic index, 0.5=Gaussian, 1=Exponential disk...
    fluxlim = 10**(-6)
    rms = 0.2*fluxlim
    notes+=' fluxlimit='+'%g'%fluxlim
    profile = 1.0
    KappaZ = 0.2 #Redshift of Kappa
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
    catflux = np.power(10,hdu[1].data['i_1400'])
    catz = hdu[1].data['redshift']
    hdu.close()
    catn = len(catra)

    ###############################Image set-up###########################
    
    fov = math.ceil(skysz*60./2.)
    if slsim==1:
        rng = 5*fov
        Extra+='_SL'
    else:
        rng = fov
    catracen = np.random.uniform(catra.min()+rng,catra.max()-rng)
    catdeccen = np.random.uniform(catdec.min()+rng,catdec.max()-rng)
    notes+=' cen='+str(catracen)+','+str(catdeccen)

    pad=5*np.sort(catsize)[np.ceil(len(catsize)*0.9)]
    rngp = rng-pad
    #selectidx = (catra>catracen-rngp)*(catra<catracen+rngp)*(catdec>catdeccen-rngp)*(catdec<catdeccen+rngp)
    selectidx = ((catra-catracen)**2+(catdec-catdeccen)**2)<rngp**2
    print "Creating Array"
    
    x=np.arange(-rng,rng,pix)
    x[np.abs(x)<0.5*pix] = 0
    #xx,yy=np.meshgrid(x,x)
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
    notes+='_n lensed sources='+str((lensed*(b1**2+b2**2<fov**2)).sum())

    ####For simple Gals
    sgals=0
    if sgals==1:
        ####Grid
        s = np.ceil((s**0.5))**2
        bs = np.floor(s**0.5)
        b1 = rngp-2*rngp*(np.arange(s)%bs)/(bs-1)
        b2 = rngp-2*rngp*np.floor(np.arange(s)/bs)/(bs-1)
        Extra += 'Grid'
        ####Simple params
        mag = np.ones(s)
        sig = 2*np.ones(s)
        Extra+='simp_mag'
        z = np.ones(s)
        component = 4*np.ones(s)


    #############################Lensing signal########################
    
    if type(Kappa) is str:
        Ktype = 3
        GammaMax=0.
        hdu=pyfits.open('Kappainputs/CFHTLenS-w1.fits')
        Kappafull = hdu[0].data
        hdu.close()
    elif Kappa==0:
        Ktype = 0
    elif len(Kappa)==2:
        Ktype = 2
        kmin,SigGamma = Kappa[0],Kappa[1]
    elif len(Kappa)==5:
        Ktype = 1
        Extra+=str(Kappa[1])+str(Kappa[2])+str(Kappa[3])+str(Kappa[4])
        amp = Kappa[0]
        x1,y1,x2,y2 = 0.001*np.array(Kappa[1:])
        
        GammaMax = amp
    else:
        print 'Unknown kappa type'
        
    ###############################Lensing Maps###########################

    print "Generating Gamma Maps"

    if Ktype==0:
        gam1map=np.zeros((szx,szx))
        gam2map=np.zeros((szx,szx))
    elif Ktype<3:
        u = np.fft.fftfreq(szx,pix)
        if Ktype==2:
            uu,vv = np.meshgrid(u,u,sparse=1)
            uu = uu[:,:szx/2+1]
            vv = vv[:,:szx/2+1]
            umax = np.abs(np.fft.fftfreq(szx,kmin)).max()+0.9*u[1]
            szk = (np.abs(u)<=umax).sum()
            SigGamma *= 0.5*np.log(2)*szx**2*szk**-1
            fgam = np.random.normal(0,SigGamma,(2,szx,int(szx/2)+1))
            fgam1 = np.zeros((szx,int(szx/2)+1),dtype=complex)+fgam[0,:,:]
            fgam2 = np.zeros((szx,int(szx/2)+1),dtype=complex)+fgam[1,:,:]
            phase = np.exp(complex(0,1)*(2*np.pi*np.random.random_sample((2,szx,int(szx/2)+1))-np.pi))
            fgam1 = np.fft.fftshift(fgam1,0)*phase[0,:,:]*(uu**2+vv**2<umax**2)
            fgam2 = np.fft.fftshift(fgam2,0)*phase[1,:,:]*(uu**2+vv**2<umax**2)
            uu,vv=2*[None]
        else:
            x1coord = int(np.round(x1/u[1]))
            y1coord = int(np.abs(np.round(y1/u[1])))
            x2coord = int(np.round(x2/u[1]))
            y2coord = int(np.abs(np.round(y2/u[1])))
            fgam1 = np.zeros((szx,int(szx/2)+1),dtype=complex)
            fgam2 = np.zeros((szx,int(szx/2)+1),dtype=complex)
            phase = np.exp(complex(0,1)*(2*np.pi*np.random.rand(2)-np.pi)*(((x1coord==0)*(y1coord==0))==0,((x2coord==0)*(y2coord==0))==0))
            fgam1[x1coord,y1coord] = szx**2*amp*phase[0]
            fgam2[x2coord,y2coord] = szx**2*amp*phase[1]
            if y1coord>0:
                fgam1 *= 0.5
            if y2coord>0:
                fgam2 *= 0.5
        gam1map = np.fft.fftshift(np.fft.irfftn(fgam1,(szx,szx)))
        gam2map = np.fft.fftshift(np.fft.irfftn(fgam2,(szx,szx)))
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

    print "Extracting Gamma Amplitudes"

    gamfac=np.power(2,-0.5)#factor of root two scales e for gamma
    if eint==0:
        gam1 = np.zeros(s)
        gam2 = np.zeros(s)
    else:
        gam1 = np.random.normal(0,eint*gamfac,s)
        gam2 = np.random.normal(0,eint*gamfac,s)
    invpix = (len(x)-1)/(x.max()-x.min())
    xcoord = np.round((b1-x[0])*invpix)
    ycoord = np.round((b2-x[0])*invpix)
    for i in range(int(s)):
        gam1[i]+= gam1map[ycoord[i],xcoord[i]]
        gam2[i]+= gam2map[ycoord[i],xcoord[i]]

    gam1map = gam1map[szx/2-szx/10:szx/2+szx/10,szx/2-szx/10:szx/2+szx/10]
    gam2map = gam2map[szx/2-szx/10:szx/2+szx/10,szx/2-szx/10:szx/2+szx/10]
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
            ordercut = 15
            cutrng = sig[i]*np.log(mag[i]/(10**-ordercut))
            indx=np.where((x>=b1[i]-cutrng-pix)*(x<=b1[i]+cutrng+pix))
            indy=np.where((x>=b2[i]-cutrng-pix)*(x<=b2[i]+cutrng+pix))
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
            temp = mag[i]*np.exp(-(temp/sig[i])**(1/(2*profile)))
            a,b,c,d=indx[0][0],indx[0][-1],indy[0][0],indy[0][-1]
            f[c:d+1,a:b+1]+=temp
##        print "source",i,"complete"
    xb,yb,temp=3*[None]

    ######Create Beam Model
    xx,yy=np.meshgrid(x,x,sparse=1)
    B = np.sinc(0.6*(xx**2+yy**2)**0.5/fov)
    xx,yy = 2*[None]
    
    ######Fourier Domain

    print "Simulating Radio obs"
    
######Numerical    
##    ukm = 3*36**2*u/(14*np.pi)
    srng = szx*fov/(2*rng)

    ##Calculate Antenna noise\ Sampling pattern
    if Plot==1:
            fsave = (B*f)[szx/2-srng:szx/2+srng,szx/2-srng:szx/2+srng]
    if Noise==0:
        f = (B*f)[szx/2-srng:szx/2+srng,szx/2-srng:szx/2+srng]
        num = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
    elif Noise==1:        
        print 'Sampling'
        ubins = np.fft.fftshift(np.fft.fftfreq(szx+1,pix))
        upix = np.gradient(ubins)[0]
        ubins -= upix/2
        ubins *= 3*36**2/(14*np.pi)
        hdu = pyfits.open('UVcov/SKA8H.fits')
        ubase = hdu[1].data['U']
        vbase = hdu[1].data['V']
        hdu.close()
        ### Select baseline subset
##        nant = ubase.size
##        ubase = ubase[range(0,nant,100)]
##        vbase = vbase[range(0,nant,100)]
        mask = np.histogram2d(ubase,vbase,ubins)[0]>0
        f *= B
        num = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))*mask
        f = None
        if slsim==1:
            print 'Gridding'
            mask = np.pad(mask[:-2,:-2],((2,0),(2,0)),mode='constant')
            mask = mask.reshape(szx/5,5,szx/5,-1).sum((1,3))
            mask[mask==0] = 1
            num = np.pad(num[:-2,:-2],((2,0),(2,0)),mode='constant')
            num = num.reshape(szx/5,5,szx/5,5).sum((1,3))/mask
    else:
        frms = 2**(-0.5)*rms*szx
        print 'Generating Noise map'
        fnoise = np.zeros((szx,szx),dtype=complex)
        posnoise = np.random.randn(szx/2-1,szx-1)+complex(0,1)*np.random.randn(szx/2-1,szx-1)
        znoise = np.random.randn(szx/2-1,3)+complex(0,1)*np.random.randn(szx/2-1,3)
        fnoise[0,0] += np.random.randn()
        fnoise[0,1:szx/2] += znoise[:,0]
        fnoise[0,szx/2] += np.random.randn()
        fnoise[0,szx/2+1:] += np.conj(znoise[:,0][::-1])
        fnoise[1:szx/2,0] += znoise[:,1]
        fnoise[1:szx/2,1:] += posnoise
        fnoise[szx/2,0] += np.random.randn()
        fnoise[szx/2,1:szx/2] += znoise[:,2]
        fnoise[szx/2,szx/2] += np.random.randn()
        fnoise[szx/2,szx/2+1:] += np.conj(znoise[:,2][::-1])
        fnoise[szx/2+1:,0] += np.conj(znoise[:,1][::-1])
        fnoise[szx/2+1:,1:] += np.conj(np.rot90(posnoise,2))
        posnoise,znoise = 2*[None]
        num = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
        f=None
        if Noise==2:
            fnoise *= frms
            fnoise *= rms/(B*np.fft.ifftn(fnoise)).std()
            if slsim==1:
                fnoise *= 0.25
            num += fnoise
            f = (B*np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(num))).real)[szx/2-srng:szx/2+srng,szx/2-srng:szx/2+srng]
            num = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))            
        if Noise==3:
            print 'Sampling'
            ubins = np.fft.fftshift(np.fft.fftfreq(szx+1,pix))
            upix = np.gradient(ubins)[0]
            ubins -= upix/2.
            ukmbins = 3*36**2*ubins/(14*np.pi)
            hdu = pyfits.open('UVcov/SKA8H.fits')
            ubase = hdu[1].data['U']
            vbase = hdu[1].data['V']
            hdu.close()
            ### Select baseline subset
##                nant = ubase.size
##                ubase = ubase[range(0,nant,100)]
##                vbase = vbase[range(0,nant,100)]
            bcnt = np.histogram2d(ubase,vbase,ukmbins)[0]
            bcnt[0,:] = 0
            bcnt[:,0] = 0
            mask = bcnt>0
            weights = np.zeros((szx,szx))
            weights[mask] = np.sqrt(1./bcnt[mask])
            weights *= mask.sum()/weights.sum()
            fnoise *= frms/np.sqrt(B.mean())
            fnoise = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(B)*np.fft.ifftn(np.fft.ifftshift(fnoise))))
            fnoise *= weights
            bcnt,weights = 2*[None]
            ### Sample
            num[mask==0] = 0
            print 'Normailze noise'
            if slsim==1:
                mask = np.pad(mask[:-2,:-2],((2,0),(2,0)),mode='constant')
                mask = mask.reshape(szx/5,5,szx/5,-1).sum((1,3))
                mask[mask==0] = 1
                xtraN = rms/np.fft.ifftn(np.fft.ifftshift(np.pad(fnoise[:-2,:-2],((2,0),(2,0)),mode='constant').reshape(szx/5,5,szx/5,5).sum((1,3))/mask)).std()
            else:
                xtraN = rms/np.fft.ifftn(np.fft.ifftshift(fnoise)).std()
            fnoise *= xtraN
            num += fnoise
            if slsim==1:
                print 'Gridding'
                num = np.pad(num[:-2,:-2],((2,0),(2,0)),mode='constant')
                num = num.reshape(szx/5,5,szx/5,5).sum((1,3))/mask
    if (Noise==0)+(Noise==2):
        num *= B[szx/2-srng:szx/2+srng,szx/2-srng:szx/2+srng]>0.5
    B = None
    f = None
    if Plot==1:
        print str((lensed*(b1**2+b2**2<fov**2)).sum()),'galaxies'
        plt.figure('True Sky')
        plt.pcolor(fsave)
        plt.colorbar()
        plt.figure('Sky')
        plt.pcolor(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(num))).real)
        plt.colorbar()
        plt.figure('Vis')
        plt.pcolor(np.log10(np.abs(num)+1))
        plt.colorbar()
        plt.show()
        exit()
    
    #############################Write data#######################

    u = np.fft.fftshift(np.fft.fftfreq(num.shape[0],pix))
    infov = b1**2+b2**2<fov**2
    lensed  = lensed[infov]
    sig = sig[infov]
    mag = mag[infov]
    b1 = b1[infov]
    b2 = b2[infov]

    
    
    print "Saving"
    Path="Sims"
    fname=str(int(skysz)),"_",str(int(60/pix)),"_",str(int(10*eint)),"_",str(int(Ktype)),Extra,".dat"
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

    upix = np.gradient(u)[0]
    kax = u[np.abs(u)<kmax+0.9*upix]
    szk = kax.size
    kx,ky = np.meshgrid(kax,kax)
    k1res=np.zeros([szk,szk],dtype=complex)
    k2res=np.zeros([szk,szk],dtype=complex)

##    #########################Do#################################

    uk1=(uu**2-vv**2)
    uk2=2*uu*vv

    Bmod = np.sinc(2*0.6*np.sqrt(b1**2+b2**2)/(pix*szu))
    
    w = np.ones(s)
    w += 1./(Bmod*mag)
        
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
    k1res = mult*np.conj(k1res)
    k2res = mult*np.conj(k2res)
    
################### Write ###################################

    wfile=open(os.path.join("Results",fname),'wb')
    np.savez(wfile,k1res=k1res,k2res=k2res,kmax=kmax,szk=szk,gam1map=gam1map,gam2map=gam2map,Kappa=Kappa,pix=pix,u=u,notes=notes)
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

    b1 = b1[lensed]
    b2 = b2[lensed]
    mag = mag[lensed]

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

    uk1=uu**2-vv**2
    uk2=2*uu*vv

    Bmod = np.sinc(2*0.6*np.sqrt(b1**2+b2**2)/(pix*szu))
    Bmod *= Bmod>=0.5
    
    w = np.ones(s)
    w += 1./(Bmod*mag)

##    print 'starting analysis'
    pg=0
    for i in range(istart,iend):
        xcen=gax[i]
        for j in range(int(szg)):
            ycen=gax[j]
            gsub=np.where((np.abs(b1-xcen)<gpix/2.)*(np.abs(b2-ycen)<gpix/2.))
            for g in gsub[0]:
                pg += w[g]*np.power(eu,complex(0,1)*b1[g])*np.power(ev,complex(0,1)*b2[g])
            gres=num*np.conj(pg)
            gam1res[i,j] = np.sum(uk1*gres)
            gam2res[i,j] = np.sum(uk2*gres)
            ngal[i,j] = gsub[0].size
            pg=0
    ngal[ngal==0] = 1
    mult = -1./ngal
    gam1res *= mult
    gam2res *= mult
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
        
