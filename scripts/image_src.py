import sys
#sys.path.append('/home/user/clip/')
from const import *
import pyfits as pf
from numba import jit
import numpy as np
from astropy.time import Time
from astropy import units as u
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt, pylab
from clip import fast_calc_uvw 
import scipy
from scipy.special import erf as erf
from scipy.fftpack import fft2 as fft2
from scipy.fftpack import ifft2 as ifft2
import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
me = casautil.tools.measures()
from joblib import Parallel, delayed
from astropy import units as u
from astropy.coordinates import SkyCoord

@jit
def fftFunc(vis, w_g, w_fac):
        w_harm = np.exp(-1j*w_g*w_fac)
        return np.fft.fftshift(fft2(np.fft.fftshift(vis[:,:]))) * w_harm
                
@jit
def ddp(dat,f,tsamp,dm=56.77118):
	nt,nf  = dat.shape
	delays = 4.15e-3 * (f/1e9)**-2.0 * dm
	delays-=np.amin(delays)
	i_delay = np.round(delays/tsamp).astype(np.int)
	print i_delay
	out = np.nan*np.zeros(dat.shape,dtype=dat.dtype)
	for i in range(nt):
		shift = i_delay[i]
		if shift>0:
			for j in range(0,nt-shift):
				dat[j,i]=dat[j+shift,i]
	return dat 
	
@jit
def dm_delay(f,dm=56.77118,f0=1530e6):
# Returns the dispervive delay relative to a reference frequency (1530 MHz default)
	return 4.15e-3 * dm*( (f/1e9)**-2.0 - (f0/1e9)**-2.0 )


@jit
def dirty_image_no_wproj(vis,u,v,w,fov,res,min_base,max_base,kernel_type='nst',fw1=2.52, fw2=1.55, alpha=2.0, msup=6):
# Take visibilities and grid them. Nearset neighbour for now.
        # fw1, fw2, and alpha are gridding kernel parameters (exp * sinc) (see White bible page 131)
        # fov is field of view in degrees
        # res is image resolution in arcsecs
        # vis is the visibilities with dims = nchans * nbaselines
        # u,v are the uv values with dims = nchans * nbaselines
        # min_base is the monimum baseline in wavelengths (max_base similarly)
        # msup is the support of the convolution kernel (use even numbers only)
        #
        #
        print 'here'
        nf,nbas=u.shape                                         # Num of baselines and num of channels
        du=1.0/(fov*np.pi/180.0)                                # uv grid resolution
        umax = 1.0/(res/3600.0*np.pi/180.0)/2.0         # Max u
        n=2*umax/du
        bl = (u**2.0+v**2.0)**0.5
        nfft = int(2**np.ceil(np.log(n)/np.log(2)))             # grid size for FFT
	print("FFT size = %d X %d"%(nfft,nfft))

        u_g  = np.arange(-du*float(nfft/2),du*float(nfft/2),du) # Initialize arrays for gridded quantities
        v_g  = np.arange(-du*float(nfft/2),du*float(nfft/2),du)
        vis_g = np.zeros((nfft,nfft),dtype=np.complex64)
        wt_g = np.zeros((nfft,nfft),dtype=np.float32)
        print "du = %f"%du

        # Precompute some gridding kernel parameter values and apodization corrections
        exp_fac = 1.0/du/fw1
        sinc_fac=np.pi/fw2/du
        offset = nfft/2


        print 'Beginning gridding'
        for i in range(vis.shape[0]):
                print i,'of',vis.shape[0]
                for j in range(vis.shape[1]):
                        #if bl[i,j]>min_base and (not np.isnan(vis[i,j].real)) and bl[i,j]<max_base:
                        if (not np.isnan(vis[i,j].real)):
                                tpu=(u[i,j]/du)
                                Iu=int(np.round(tpu))
                                tpv=(v[i,j]/du)
                                Iv=int(np.round(tpv))
				if kernel_type=='sph':
                                	for jj in range(Iv-msup/2+1,Iv+msup/2+1):
                                       		for ii in range(Iu-msup/2+1, Iu+msup/2+1):      
                                               		disp = du * ( (tpu-ii)**2.0 + (tpv-jj)**2.0  )**0.5
                                               		ker = ((np.exp(-(disp*exp_fac)))**alpha  * np.sin(disp*sinc_fac)/(disp*sinc_fac))               
                                               		vis_g[jj+offset,ii+offset]+= vis[i,j]*ker
                                               		wt_g[jj+offset,ii+offset]+=ker
				else:
	                                vis_g[Iv+offset,Iu+offset]+=vis[i,j]    # This is nearest neighbour gridding
       	  	                        wt_g[Iv+offset,Iu+offset]+=1.0

                                tpu=(-u[i,j]/du)
                                Iu=int(np.round(tpu))
                                tpv=(-v[i,j]/du)
                                Iv=int(np.round(tpv))
				if kernel_type=='sph':
                                	for jj in range(Iv-msup/2+1,Iv+msup/2+1):
                                       		for ii in range(Iu-msup/2+1, Iu+msup/2+1):
                                               		disp = du * ( (tpu-ii)**2.0 + (tpv-jj)**2.0  )**0.5
                                               		ker = ((np.exp(-(disp*exp_fac)))**alpha  * np.sin(disp*sinc_fac)/(disp*sinc_fac))               
                                               		vis_g[jj+offset,ii+offset]+= np.conjugate(vis[i,j])*ker
                                               		wt_g[jj+offset,ii+offset]+=ker
				else:
                                	vis_g[Iv+offset,Iu+offset]+=np.conjugate(vis[i,j])
                                	wt_g[Iv+offset,Iu+offset]+=1.0


        for i in range(wt_g.shape[0]):
                for j in range(wt_g.shape[1]):
                        if wt_g[i,j]>0:
                                vis_g[i,j]/=wt_g[i,j]
                                wt_g[i,j]=1.0
                                bl = (u_g[i]**2.0+u_g[j]**2.0)**0.5
                                taper = (1.0+erf( (bl-min_base)/du ))/2.0 * (1.0-erf( (bl-max_base)/du))/2.0
                                vis_g[i,j]*=taper
                                wt_g[i,j]*=taper

        norm =np.sum(wt_g)
        if norm==0:
                print "No visibilities to grid!"; exit(1);
        vis_g/=norm
        wt_g/=norm

        print "beginning transforming ..."
        im = np.fft.fftshift(fft2(np.fft.fftshift(vis_g)))
        psf = np.fft.fftshift(fft2(np.fft.fftshift(wt_g)))
        im/=np.amax(im.real)
	psf/=np.amax(psf.real)
	return u_g,im,psf




#@jit
def dirty_image(vis,u,v,w,fov,res,min_base,max_base):
	# Take visibilities, grid them, and make a dirty image
	# fov is field of view in degrees (default is 6.2 deg which is the null-null width for a 4.5 m aperture at 1500 MHz)
	# res is image resolution in arcsecs (default is lambda/umax/4 for lambda = 3e8/1.5e9 and umax =1200 m)
	# vis is the visibilities with dims = nchans * nbaselines
	# u,v,w are the uvw values with dims = nchans * nbaselines
	# min_base is the monimum baseline in wavelengths (max_base similarly)
	# msup is the support of the convolution kernel (use even numbers only)
	#
	#
	wmax = np.amax(np.absolute(w))					# The largest w term
	tp = wmax*2.0*np.pi*(1.0-np.cos(fov/2.0*np.pi/180.0)) * 10.0	# 2pi*(wmax-wmin)*(1-n) is the max w-term phase; sample this say 10 times at-least
	Nw= int((tp/2) * 2+1)						# This maks sure Nw is an odd number
	w_g = np.linspace(-wmax,wmax,Nw)
	dw=w_g[1]-w_g[0]

	nf,nbas=u.shape						# Num of baselines and num of channels
	du=1.0/(fov*np.pi/180.0)				# uv grid resolution
	umax = 1.0/(res/3600.0*np.pi/180.0)/2.0			# Max u
	n=2*umax/du
	bl = (u**2.0+v**2.0)**0.5
	nfft = int(2**np.ceil(np.log(n)/np.log(2)))		# grid size for FFT
	print("FFT size = %d X %d"%(nfft,nfft))
	offset = nfft/2						# Array offsets to get u=0 at i=nfft/2

	u_g  = np.arange(-du*float(nfft/2),du*float(nfft/2),du)	# Initialize arrays for gridded quantities
	v_g  = np.arange(-du*float(nfft/2),du*float(nfft/2),du)
	vis_g = np.zeros((Nw,nfft,nfft),dtype=np.complex64)
	wt_g = np.zeros((Nw,nfft,nfft),dtype=np.float32)
	print "du = %f"%du



	dl = 1.0/(np.amax(u_g)-np.amin(u_g))
	l_g  = np.arange(-dl*nfft/2,dl*nfft/2,dl)	# Initialize arrays for gridded quantities


	for i in range(vis.shape[0]):			#For Each channel
		for j in range(vis.shape[1]):		# For each baseline
			if bl[i,j]>min_base and (not np.isnan(vis[i,j].real)) and bl[i,j]<max_base:	# Vsibility selection filter
				wslice=int(np.round(w[i,j]/dw))+((Nw-1)/2)				# w-term slice number
				tpu=(u[i,j]/du)
				Iu=int(np.floor(tpu))
				tpv=(v[i,j]/du)
				Iv=int(np.floor(tpv))
				vis_g[wslice,Iv+offset,Iu+offset]+=vis[i,j] 	# This is nearest neighbour gridding
				wt_g[wslice,Iv+offset,Iu+offset]+=1.0


	w_fac = np.zeros((nfft,nfft),dtype=np.float32)
	for i in range(nfft):
		for j in range(nfft):
			w_fac[i,j] = 2.0*np.pi*( ( 1.0-l_g[i]**2.0-l_g[j]**2.0  )**0.5 - 1.0 )

	im=np.zeros((nfft,nfft),dtype=np.complex64)
	psf=np.zeros((nfft,nfft),dtype=np.complex64)

        # parallel version
        print 'Starting parallel invert with %d w-planes'%(Nw)
        w_slices = Parallel(n_jobs=16)(delayed(fftFunc)(vis_g[i,:,:],w_g[i],w_fac) for i in range(Nw))
        w_slices_psf = Parallel(n_jobs=16)(delayed(fftFunc)(wt_g[i,:,:],w_g[i],w_fac) for i in range(Nw))
        im = np.sum(np.asarray(w_slices),axis=0)
        psf = np.sum(np.asarray(w_slices_psf),axis=0)
        print 'Finished!'
        
	im/=np.amax(im.real)
	psf/=np.amax(psf.real)	
	return u_g,im,psf

@jit
def norm(d):
	d/=np.absolute(d)

@jit
def interp_phase(d,bsf=30):
	maxf=np.amax(FREQ2)
	f_cal=FREQ1[bsf/2::bsf]/1e9
	dr=np.zeros((d.shape[0],NF2),dtype=np.float32)
	di=np.zeros((d.shape[0],NF2),dtype=np.float32)
	for i in range(d.shape[0]):
		dr[i,:]=np.interp(maxf-FREQ2/1e9,maxf-f_cal,d[i,:].real)
		di[i,:]=np.interp(maxf-FREQ2/1e9,maxf-f_cal,d[i,:].imag)
	
		plt.plot(f_cal,np.angle(d[i,:]),'o',label='solution')
		plt.plot(FREQ2/1e9,np.angle(dr[i,:]+1j*di[i,:]),'-',label='interpolation')
		plt.xlabel('Freq [GHz]'); plt.ylabel('Phase [radian]')
		plt.legend(frameon=False)
		plt.title('Antenna number %d'%i)
		plt.tight_layout()
		plt.savefig('temp/%d.png'%i)
		plt.close()
	
	return dr+1j*di




def calc_image_uvw(vis_fname="test.fits", image_centre=[ 1.4596748494230258, 0.38422453855292943], savez=True):
        print "will compute uvw"
        f=FREQ2
        h = pf.open(vis_fname)[1].header
        tau = h['TSAMP']
        toa = h['MJD']
	dispersion_measure = h['DM']
        print 'Have DM: ',dispersion_measure
        antennas = h['ANTENNAS'].split('-')
        tp=np.loadtxt('/home/user/IMAGING_FULL/antpos_ITRF.txt')
        nants = len(antennas)
        blen=np.zeros((nants*(nants-1)/2,3),dtype=np.float32)

        u=np.zeros((NF2,nants*(nants-1)/2))
        v=np.zeros((NF2,nants*(nants-1)/2))
        w=np.zeros((NF2,nants*(nants-1)/2))
        wcorr=np.zeros((NF2,nants*(nants-1)/2))	# This is the w correction for rotation of the phase centre

        iiter=0
        for i in range(nants-1):
                for j in range(i+1,nants):
                        a1=int(antennas[i])-1; a2=int(antennas[j])-1;
                        blen[iiter,:]=np.array(tp[a1,1:]-tp[a2,1:])
                        iiter+=1

        delays = dm_delay(f,dm=dispersion_measure)
        t_mjd = toa + delays/SinD		# SinD is seconds in a day defined in const.py
						# t_mjd has the arrival time of pulses in each channel
        print "TOA = "+str(toa)
        print "Max disp delay = "+str(np.max(delays))

        qa = casautil.tools.quanta()
        me = casautil.tools.measures()
        #me.doframe(me.observatory('DSA-10'))            # Observatory (will look for entry in casacore measures:directory (see ~/.casarc)
        me.doframe(me.observatory('OVRO_MMA'))            # Observatory (will look for entry in casacore measures:directory (see ~/.casarc)
        me.doframe(me.epoch('utc',qa.quantity(REF_EPOCH,'d')))
        b=me.direction('HADEC', qa.quantity(0.0,'rad'),qa.quantity(POINTING,'rad'))
        azel=me.measure(b,'AZEL')
        me.doframe(me.epoch('utc',qa.quantity(toa,'d')))
        radec=me.measure(azel,'J2000')
        lst=radec['m0']['value']
        dec=radec['m1']['value']
        
        lam=C/f
        for i in range(nants*(nants-1)/2):
                tp4,tp5,tp6=fast_calc_uvw(blen[i,:],t_mjd)#,pointing=POINTING,src=[lst,dec])
                tp1,tp2,tp3=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING,src=[image_centre[0],image_centre[1]])
                u[:,i]=tp1/lam
                v[:,i]=tp2/lam
                w[:,i]=tp3/lam
                wcorr[:,i]=(tp3-tp6)/lam

	#if savez:
	#uv_fname = "%s_uvw.npz"%vis_fname[:-4]
	#np.savez("test_withCasa.npz",u=u,v=v,w=w,wcorr=wcorr,f=f,t_mjd=t_mjd)

        
	return u,v,w,wcorr



def applycal(cal_fname_x, cal_fname_y, vis_fname,flag_ants, flag_chans, burst_loc, burst_nsamp):

	tp=np.load(cal_fname_x)
	anames_x=list(tp['aname'])
	taus_x=tp['ant_delays']
	g_x = tp['gains']
	bsf_x=tp['freq_cadence']	
	norm(g_x)                       # Normalize the gains (take only phases)
	#g_x=np.nanmean(g_x[:,g_x.shape[1]/2-2:g_x.shape[1]/2+2,:],axis=1)
        #g_x = g_x - g_x + (1.+0.*1j)

	tp=np.load(cal_fname_y)
	anames_y=list(tp['aname'])
	taus_y=tp['ant_delays']
	g_y=tp['gains']
	bsf_y=tp['freq_cadence']
	norm(g_y)
	#g_y=np.nanmean(g_y[:,g_y.shape[1]/2-2:g_y.shape[1]/2+2,:],axis=1)
	#g_y = g_y - g_y + (1.+0.*1j)

	g_x_interp = interp_phase(g_x,bsf_x)
	g_y_interp = interp_phase(g_y,bsf_y)

        f=FREQ2
        h = pf.open(vis_fname)[1].header
        antennas = h['ANTENNAS'].split('-')
        nants = len(antennas)

        bline_delay_x=np.zeros((nants*(nants-1)/2),dtype=np.float32)
        bline_delay_y=np.zeros((nants*(nants-1)/2),dtype=np.float32)
        phase_x=np.zeros((nants*(nants-1)/2,NF2),dtype=np.float32)
        phase_y=np.zeros((nants*(nants-1)/2,NF2),dtype=np.float32)

        iiter=0
        for i in range(nants-1):
                for j in range(i+1,nants):
                        a1=int(antennas[i])-1; a2=int(antennas[j])-1;

                        t1=taus_x[a1]
                        t2=taus_x[a2]
                        g1=g_x_interp[a1,:]
                        g2=g_x_interp[a2,:]
                        phase_x[iiter,:] = np.angle(g1*np.conjugate(g2))
                        bline_delay_x[iiter]=t1-t2

                        t1=taus_y[a1]
                        t2=taus_y[a2]
                        bline_delay_y[iiter]=t1-t2
                        g1=g_y_interp[a1,:]
                        g2=g_y_interp[a2,:]
                        phase_y[iiter,:] = np.angle(g1*np.conjugate(g2))
                        iiter+=1


	d=pf.open(vis_fname)[1].data['Data'].astype(np.float32)
	header = pf.open(vis_fname)[1].header
	nt = (header['NAXIS2']-1)/(2048*200)
	print("There are %d time-samples in visibility data"%nt)
	d1=d[:nt*2048*200]; 
	dat = np.reshape(d1,newshape=(nt,2048,200)); 
	ac = dat[:,F_START:F_END,180:]

	cc = np.mean(dat[burst_loc:burst_loc+burst_nsamp,F_START:F_END,:180],axis=0);
        l1 = burst_loc*1.
        l2 = 1.*(dat.shape[0]-(burst_loc+burst_nsamp))
        #cc_noise = (l1/(l1+l2))*np.mean(dat[:burst_loc,F_START:F_END,:180],axis=0)+(l2/(l1+l2))*np.mean(dat[burst_loc+burst_nsamp:,F_START:F_END,:180],axis=0)
        #cc -= cc_noise
	

	cc1=cc[:,::2]+1j*cc[:,1::2]
	cc1[np.where(cc1.real==0.0)]=np.nan+1j*np.nan
	cc_xx=cc1[:,::2]; 
	cc_yy=cc1[:,1::2]
	#cc_xx[flag_chans,:]=np.nan
	#cc_yy[flag_chans,:]=np.nan

	# Flag antennas
	iiter=0
	for i in range(len(antennas)-1):
	        for j in range(i+1,len(antennas)):
        	        if (antennas[i] in flag_ants) or (antennas[j] in flag_ants):
                	        cc[:,4*iiter:4*(iiter+1)] = np.nan
        	        iiter+=1

	# Following is for normalization using autocorrrelations (optional; uncomment if necessary)
	#	If using make sure to multiple with cal_x and cal_y later on while applying bandpass calibration
	ac[np.where(ac==0)]=np.nan
	ac_norm_x = np.nanmedian(ac[:,:,0::2],axis=0)
	ac_norm_y = np.nanmedian(ac[:,:,1::2],axis=0)
	cal_x = np.zeros((NF2,nants*(nants-1)/2),dtype=np.float32)
	cal_y = np.zeros((NF2,nants*(nants-1)/2),dtype=np.float32)
	iiter=0
	for i in range(nants-1):
	        for j in range(i+1,nants):
	                cal_x[:,iiter] = (ac_norm_x[:,i]*ac_norm_x[:,j])**-0.5
	                cal_y[:,iiter] = (ac_norm_y[:,i]*ac_norm_y[:,j])**-0.5
	                iiter+=1

	vis = np.zeros((NF2,nants*(nants-1)/2),dtype=np.complex64)

	for i in range(nants*(nants-1)/2):
	        fac_x=np.exp(-1j*PI2*f*bline_delay_x[i]-1j*phase_x[i,:]).astype(np.complex64)
	        fac_y=np.exp(-1j*PI2*f*bline_delay_y[i]-1j*phase_y[i,:]).astype(np.complex64)
	        vis[:,i] =( (cc[:,4*i]+1j*cc[:,4*i+1])*fac_x*cal_x[:,i]  + (cc[:,4*i+2]+1j*cc[:,4*i+3])* fac_y*cal_y[:,i]  ) /2.0
                #vis[:,i] = (cc[:,4*i]+1j*cc[:,4*i+1])*fac_x*cal_x[:,i] 
                #vis[:,i] =( (cc[:,4*i]+1j*cc[:,4*i+1])*fac_x  + (cc[:,4*i+2]+1j*cc[:,4*i+3])* fac_y  ) /2.0

	return vis 


############################################################################
#
#		MAIN BEGINS HERE
#
############################################################################

# arguments: input fits name, output dir, RA/DEC (rad)
vis_fname = sys.argv[1]
odir = sys.argv[2]
oname = sys.argv[3]

cco = SkyCoord(sys.argv[4],sys.argv[5],unit=(u.hourangle,u.deg))
ra = cco.ra.radian
dec = cco.dec.radian

fov=0.5 # deg
res=5.0 # arcsec
min_base=170.0 #lambda
max_base=1000000.0
burst_nsamp=1
cal_fname_x = odir+"/Acal.npz"
cal_fname_y = odir+"/Bcal.npz"
flag_ants=['2','8']
flag_chans=np.concatenate((np.arange(210,215),np.arange(710,730)))#range(1370,1400)  #channels from 300
object_name = "FRB-CAND"
opfname_prefix="frb"
time = Time(pf.open(vis_fname)[1].header['MJD'],format='mjd')
#ra = time.sidereal_time('mean',-118.2834*u.deg).value*(np.pi/180.)*15.
#dec = 73.6*np.pi/180.
image_centre = [ra, dec]
print ra, dec

# find burst location
alld = np.asarray(pf.open(vis_fname)[1].data).astype('float')
alld = alld[0:len(alld)-1]
lalld = len(alld)
nt = lalld/200/2048
alld=alld.reshape((nt,2048,200))[:,200:1848,:]
tts = np.zeros(nt)
t1 = np.zeros(nt)
t2 = np.zeros(nt)
t3 = np.zeros(nt)
print 'Have nt: ',nt
for i in np.arange(5):
        ts = np.mean(alld,axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        tts += ts/5.
        
burst_loc = np.where(tts==np.max(tts))[0][0]
burst_loc = int(sys.argv[6])
print 'Imaging at location',burst_loc



# SNR in TS
S2 = (np.max(tts)-np.mean(tts))/np.std(tts)

# Calculate the u,v,w and delta_delay for image centre rotation
u,v,w,wcorr = calc_image_uvw(vis_fname=vis_fname, image_centre=image_centre, savez=True)

# Read visibilities and apply calibration solutions to them
vis = applycal(cal_fname_x=cal_fname_x,cal_fname_y=cal_fname_y,vis_fname=vis_fname, flag_ants=flag_ants, flag_chans=flag_chans, burst_loc=burst_loc,burst_nsamp=burst_nsamp)

# Phase rotate calibrated visibilities to the image centre
w_term = np.exp(-1j*2.0*np.pi*wcorr).astype(np.complex64)	# Multiplying by w_term shifts the phase centre to image_centre
vis*=w_term
u_g,im,psf = dirty_image(vis=vis,u=u,v=v,w=w,fov=fov,res=res,min_base=min_base,max_base=max_base)

# Figure out some header keywords
n=len(u_g)
toa = pf.open(vis_fname)[1].header['MJD']
dtheta  = 1.0/( (u_g[1]-u_g[0]) * float(n)) * 180.0/np.pi

# Write results into FITS files
hdr = pf.Header([('SIMPLE',True),('BITPIX',-32),('ORIGIN','DSA-10 CLIP'), ('AUTHOR','H.K.VEDANTHAM'), ('BTYPE','FLUX-DENSITY'), ('BUNIT','JY/BEAM'), ('OBJECT',object_name), ('POL','I'), ('NAXIS',2), ('NAXIS1',n), ('NAXIS2',n), ('IMTYPE','DIRTY'), ('EPOCH',2000.0), ('EQUINOX',2000.0), ('MJD',toa), ('CTYPE1','RA---SIN'), ('CTYPE2','DEC--SIN'), ('CUNIT1','DEGREE'), ('CUNIT2','DEGREE'), ('CRVAL1',image_centre[0]*180.0/np.pi), ('CRVAL2',image_centre[1]*180.0/np.pi), ('CRPIX1',n/2), ('CRPIX2',n/2), ('CDELT1',-dtheta), ('CDELT2',dtheta)])

pf.writeto(odir+'/'+oname+'.fits',np.fliplr(im.real),hdr,clobber=True)
