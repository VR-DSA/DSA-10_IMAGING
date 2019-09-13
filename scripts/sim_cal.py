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

def fftFunc_noW(vis):
        return np.fft.fftshift(fft2(np.fft.fftshift(vis[:,:])))

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



def dirty_image_synthesis(vis,u,v,w,fov,res,min_base,max_base):
	# Take visibilities, grid them, and make a dirty image
	# fov is field of view in degrees (default is 6.2 deg which is the null-null width for a 4.5 m aperture at 1500 MHz)
	# res is image resolution in arcsecs (default is lambda/umax/4 for lambda = 3e8/1.5e9 and umax =1200 m)
	# vis is the visibilities with dims = nbaselines * ntimes * nchans
	# u,v,w are the uvw values with dims = nbaselines * ntimes * nchans
	# min_base is the monimum baseline in wavelengths (max_base similarly)
	# msup is the support of the convolution kernel (use even numbers only)
	#
	#
	wmax = np.amax(np.absolute(w))					# The largest w term
        print 'MAX BASEL',np.amax(np.absolute(u)),np.amax(np.absolute(v)),wmax
	tp = wmax*2.0*np.pi*(1.0-np.cos(fov/2.0*np.pi/180.0)) * 10.0	# 2pi*(wmax-wmin)*(1-n) is the max w-term phase; sample this say 10 times at-least
	Nw= int((tp/2) * 2+1)						# This maks sure Nw is an odd number
	w_g = np.linspace(-wmax,wmax,Nw)
	dw=w_g[1]-w_g[0]

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


        # full gridding
	for i in range(2):			# For each time sample
                print 'Gridded ',i+1,' of ',vis.shape[0],' time samples'
		for j in range(vis.shape[1]):		# For each channel
			for k in range(vis.shape[2]):	# For each antenna pair
				if bl[i,j,k]>min_base and (not np.isnan(vis[i,j,k].real)) and bl[i,j,k]<max_base:	# Vsibility selection filter
					wslice=int(np.round(w[i,j,k]/dw))+((Nw-1)/2)				# w-term slice number
					tpu=(u[i,j,k]/du)
					Iu=int(np.floor(tpu))
					tpv=(v[i,j,k]/du)
					Iv=int(np.floor(tpv))
					vis_g[wslice,Iv+offset,Iu+offset]+=vis[i,j,k] 	# This is nearest neighbour gridding
					wt_g[wslice,Iv+offset,Iu+offset]+=1.0
                                        

        # gridding assuming single uv for observation

        #print 'GRIDDING'
        #midI = int(vis.shape[0]/2)
        #for j in range(vis.shape[1]):		# For each channel
	#	for k in range(vis.shape[2]):	# For each antenna pair
	#		if bl[midI,j,k]>min_base and bl[midI,j,k]<max_base:	# Vsibility selection filter
	#			wslice=int(np.round(w[midI,j,k]/dw))+((Nw-1)/2)				# w-term slice number
	#			tpu=(u[midI,j,k]/du)
	#			Iu=int(np.floor(tpu))
	#			tpv=(v[midI,j,k]/du)
	#			Iv=int(np.floor(tpv))
        #                        for i in range(vis.shape[0]):
        #                                if (not np.isnan(vis[i,j,k].real)):
	#                                        vis_g[wslice,Iv+offset,Iu+offset]+=vis[i,j,k] 	# This is nearest neighbour gridding
	#			                wt_g[wslice,Iv+offset,Iu+offset]+=1.0


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

        #vv = np.sum(vis_g,axis=0)
        #im = fftFunc_noW(vv)
        #vv = np.sum(wt_g,axis=0)
        #psf = fftFunc_noW(vv)
        
	im/=np.amax(im.real)
	psf/=np.amax(psf.real)	
	return u_g,im,psf









@jit
def norm(d):
	d/=np.absolute(d)

@jit
def interp_phase(d,bsf=30):
	maxf=np.amax(FREQ1)
	f_cal=FREQ1[bsf/2::bsf]/1e9
	dr=np.zeros((d.shape[0],NF1),dtype=np.float32)
	di=np.zeros((d.shape[0],NF1),dtype=np.float32)
	for i in range(d.shape[0]):
		dr[i,:]=np.interp(maxf-FREQ1/1e9,maxf-f_cal,d[i,:].real)
		di[i,:]=np.interp(maxf-FREQ1/1e9,maxf-f_cal,d[i,:].imag)
	'''
		plt.plot(f_cal,np.angle(d[i,:]),'o',label='solution')
		plt.plot(FREQ1/1e9,np.angle(dr[i,:]+1j*di[i,:]),'-',label='interpolation')
		plt.xlabel('Freq [GHz]'); plt.ylabel('Phase [radian]')
		plt.legend(frameon=False)
		plt.title('Antenna number %d'%i)
		plt.tight_layout()
		plt.savefig('temp/%d.png'%i)
		plt.close()
	'''
	return dr+1j*di




def calc_image_uvw(vis_fname="output.npz", image_centre = [0.,0.], savez=True, act_image_centre = None):
        print "will compute uvw"
        f=FREQ1
        h = np.load(vis_fname)
        tau = h['TSAMP']
        toa = h['MJD']
        antennas = str(h['ANTENNAS']).split('-')
        tp=np.loadtxt('/home/user/IMAGING_FULL/antpos_ITRF.txt')
        nants = len(antennas)
        blen=np.zeros((nants*(nants-1)/2,3),dtype=np.float32)

        u=np.zeros((len(toa),NF1,nants*(nants-1)/2))
        v=np.zeros((len(toa),NF1,nants*(nants-1)/2))
        w=np.zeros((len(toa),NF1,nants*(nants-1)/2))
        wcorr=np.zeros((len(toa),NF1,nants*(nants-1)/2))	# This is the w correction for rotation of the phase centre

        iiter=0
        for i in range(nants-1):
                for j in range(i+1,nants):
                        a1=int(antennas[i])-1; a2=int(antennas[j])-1;
                        blen[iiter,:]=np.array(tp[a1,1:]-tp[a2,1:])
                        iiter+=1


        t_mjd = toa # has shape of time-samples                                                                         
        
        # do uvw calculation
        lam=C/f
        for i in range(nants*(nants-1)/2):

                if act_image_centre is not None:
                        tp4,tp5,tp6=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING,src=[act_image_centre[0],act_image_centre[1]])
                else:
                        tp4,tp5,tp6=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING)
                        
                tp1,tp2,tp3=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING,src=[image_centre[0],image_centre[1]])
                
                for j in range(len(toa)):
                        u[j,:,i]=tp1[j]/lam
                        v[j,:,i]=tp2[j]/lam
                        w[j,:,i]=tp3[j]/lam
                        wcorr[j,:,i]=(tp3[j]-tp6[j])/lam
                        
        
	return u,v,w,wcorr



def applycal(cal_fname_x, cal_fname_y, vis_fname,flag_ants,doCal=True):

        f=FREQ1
        h = np.load(vis_fname)
        antennas = str(h['ANTENNAS']).split('-')
        nants = len(antennas)
        nt = h['NAXIS2']

        
        # fill in vis
	vis = np.zeros((nt,NF1,nants*(nants-1)/2),dtype=np.complex64)

	for i in range(nants*(nants-1)/2):
                vis[:,:,i] = 1.+0.*1j
        
	# Flag antennas
	iiter=0
	for i in range(len(antennas)-1):
	        for j in range(i+1,len(antennas)):
        	        if (antennas[i] in flag_ants) or (antennas[j] in flag_ants):
                                vis[:,:,iiter] = np.nan+1j*np.nan

                        iiter+=1

	return vis 


############################################################################
#
#		MAIN BEGINS HERE
#
############################################################################

# arguments: input cal npz name, output dir, RA/DEC (rad)
vis_fname = sys.argv[1]
odir = sys.argv[2]
#cco = SkyCoord(sys.argv[3],sys.argv[4],unit=(u.hourangle,u.deg))
#ra = cco.ra.radian
#dec = cco.dec.radian

cco_src = SkyCoord("12:00:19.21","+73:00:45.7",unit=(u.hourangle,u.deg))
ra_src = cco_src.ra.radian
dec_src = cco_src.dec.radian

fov=6.0 # deg
res=15 # arcsec
toa = np.mean(np.load(vis_fname)['MJD'])
min_base=170.0 #lambda
max_base=100000.0
cal_fname_x = odir+"/Acal.npz"
cal_fname_y = odir+"/Bcal.npz"
flag_ants=[]
object_name = "CAL"
opfname_prefix="cal"
time = Time(toa,format='mjd')
ra = time.sidereal_time('mean',-118.2834*u.deg).value*(np.pi/180.)*15.
dec = 73.6*np.pi/180. 
image_centre = [ra, dec]
doCal=True
print ra, dec


# Calculate the u,v,w and delta_delay for image centre rotation
u_src,v_src,w_src,wcorr_src = calc_image_uvw(vis_fname=vis_fname, image_centre=image_centre, savez=False, act_image_centre=[ra_src, dec_src])
u,v,w,wcorr = calc_image_uvw(vis_fname=vis_fname, image_centre=image_centre, savez=False)

# Read visibilities and apply calibration solutions to them
vis = applycal(cal_fname_x=cal_fname_x,cal_fname_y=cal_fname_y,vis_fname=vis_fname, flag_ants=flag_ants, doCal=doCal)

# Phase rotate calibrated visibilities to the image centre
w_term_src = np.exp(-1j*2.0*np.pi*wcorr_src).astype(np.complex64)
w_term = np.exp(-1j*2.0*np.pi*wcorr).astype(np.complex64)	# Multiplying by w_term shifts the phase centre to image_centre
vis*=w_term_src
#vis*=w_term

# form dirty image
vis /= np.abs(vis)
u_g,im,psf = dirty_image_synthesis(vis=vis,u=u,v=v,w=w,fov=fov,res=res,min_base=min_base,max_base=max_base)

# Figure out some header keywords
n=len(u_g)
dtheta  = 1.0/( (u_g[1]-u_g[0]) * float(n)) * 180.0/np.pi

# Write results into FITS files
hdr = pf.Header([('SIMPLE',True),('BITPIX',-32),('ORIGIN','DSA-10 CLIP'), ('AUTHOR','H.K.VEDANTHAM'), ('BTYPE','FLUX-DENSITY'), ('BUNIT','JY/BEAM'), ('OBJECT',object_name), ('POL','I'), ('NAXIS',2), ('NAXIS1',n), ('NAXIS2',n), ('IMTYPE','DIRTY'), ('EPOCH',2000.0), ('EQUINOX',2000.0), ('MJD',toa), ('CTYPE1','RA---SIN'), ('CTYPE2','DEC--SIN'), ('CUNIT1','DEGREE'), ('CUNIT2','DEGREE'), ('CRVAL1',image_centre[0]*180.0/np.pi), ('CRVAL2',image_centre[1]*180.0/np.pi), ('CRPIX1',n/2), ('CRPIX2',n/2), ('CDELT1',-dtheta), ('CDELT2',dtheta)])

pf.writeto(odir+'/synthesis.fits',np.fliplr(im.real),hdr,clobber=True)

hdr = pf.Header([('SIMPLE',True),('BITPIX',-32),('ORIGIN','DSA-10 CLIP'), ('AUTHOR','H.K.VEDANTHAM'), ('BTYPE','FLUX-DENSITY'), ('BUNIT','JY/BEAM'), ('OBJECT',object_name), ('POL','I'), ('NAXIS',2), ('NAXIS1',n), ('NAXIS2',n), ('IMTYPE','PSF'), ('EPOCH',2000.0), ('EQUINOX',2000.0), ('MJD',toa), ('CTYPE1','RA---SIN'), ('CTYPE2','DEC--SIN'), ('CUNIT1','DEGREE'), ('CUNIT2','DEGREE'), ('CRVAL1',image_centre[0]*180.0/np.pi), ('CRVAL2',image_centre[1]*180.0/np.pi), ('CRPIX1',n/2), ('CRPIX2',n/2), ('CDELT1',-dtheta), ('CDELT2',dtheta)])

pf.writeto(odir+'/synthesis_psf.fits',np.fliplr(psf.real),hdr,clobber=True)


