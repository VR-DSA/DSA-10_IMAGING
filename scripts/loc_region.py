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
from scipy import stats

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
	return 4.149e-3 * dm*( (f/1e9)**-2.0 - (f0/1e9)**-2.0 )


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
        f=FREQ2
        h = pf.open(vis_fname)[1].header
        tau = h['TSAMP']
        toa = h['MJD']
	dispersion_measure = h['DM']
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

        lam=C/f
        for i in range(nants*(nants-1)/2):
                tp4,tp5,tp6=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING)
                tp1,tp2,tp3=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING,src=[image_centre[0],image_centre[1]])
                u[:,i]=tp1/lam
                v[:,i]=tp2/lam
                w[:,i]=tp3/lam
                wcorr[:,i]=(tp3-tp6)/lam

        
	return u,v,w,wcorr



def applycal(cal_fname_x, cal_fname_y, vis_fname, burst_loc, burst_nsamp, flag_ants=[]):

	tp=np.load(cal_fname_x)
	anames_x=list(tp['aname'])
	taus_x=tp['ant_delays']
	g_x = tp['gains']
	bsf_x=tp['freq_cadence']
	norm(g_x)                       # Normalize the gains (take only phases)

	tp=np.load(cal_fname_y)
	anames_y=list(tp['aname'])
	taus_y=tp['ant_delays']
	g_y=tp['gains']
	bsf_y=tp['freq_cadence']
        norm(g_y)

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
	

	cc1=cc[:,::2]+1j*cc[:,1::2]
	cc1[np.where(cc1.real==0.0)]=np.nan+1j*np.nan
	cc_xx=cc1[:,::2]; 
	cc_yy=cc1[:,1::2]

	vis = np.zeros((NF2,nants*(nants-1)/2),dtype=np.complex64)

        
	# Flag antennas and baselines
	iiter=0
	for i in range(len(antennas)-1):
	        for j in range(i+1,len(antennas)):
        	        if (antennas[i] in flag_ants) or (antennas[j] in flag_ants):
                	        cc[:,4*iiter:4*(iiter+1)] = np.nan
                        
        	        iiter+=1


        # flag channels
        #chans = np.concatenate((np.arange(0,550),np.arange(1125,1250)))
        #print chans
        #cc[chans,:] = np.nan
                        
        iiter=0
        for i in range(nants-1):
                for j in range(i+1,nants):
                        a1=int(antennas[i]); a2=int(antennas[j]);

                        fac_x=np.exp(-1j*PI2*f*bline_delay_x[iiter]-1j*phase_x[iiter,:]).astype(np.complex64)
	                fac_y=np.exp(-1j*PI2*f*bline_delay_y[iiter]-1j*phase_y[iiter,:]).astype(np.complex64)
	                vis[:,iiter] =( (cc[:,4*iiter]+1j*cc[:,4*iiter+1])*fac_x  + (cc[:,4*iiter+2]+1j*cc[:,4*iiter+3])*fac_y  ) /2.0
                        
                        iiter+=1
                
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

burst_nsamp=4
burst_loc=118
flag_ants = ['2','8']
min_base = 200. # lambda
cal_fname_x = odir+"/Acal_offset.npz"
cal_fname_y = odir+"/Bcal_offset.npz"
object_name = "FRB-CAND"
opfname_prefix="frb"
toa = pf.open(vis_fname)[1].header['MJD']
time = Time(toa,format='mjd')
#ra = time.sidereal_time('mean',-118.2834*u.deg).value*(np.pi/180.)*15.
#dec = 73.6*np.pi/180.
image_centre = [ra, dec]

# Calculate the u,v,w and delta_delay for image centre rotation
u,v,w,wcorr = calc_image_uvw(vis_fname=vis_fname, image_centre=image_centre, savez=False)

# Read visibilities and apply calibration solutions to them
vis = applycal(cal_fname_x=cal_fname_x,cal_fname_y=cal_fname_y,vis_fname=vis_fname, burst_loc=burst_loc,burst_nsamp=burst_nsamp,flag_ants=flag_ants)
vis /= np.abs(vis)
vis[np.sqrt(u**2.+v**2.)<min_base] = np.nan+1j*np.nan

# Phase rotate calibrated visibilities to the image centre
w_term = np.exp(-1j*2.0*np.pi*wcorr).astype(np.complex64)	# Multiplying by w_term shifts the phase centre to image_centre
vis_ref = vis*w_term
sig2 = 0.5*(np.nanstd(vis_ref.imag)**2.+np.nanstd(vis_ref.real)**2.)
print 'Have STD:',np.sqrt(sig2)
rmn = np.nanmean(vis_ref.real)
imn = np.nanmean(vis_ref.imag)
print 'Have MNs:',rmn,imn

# identify noisy baselines
h = pf.open(vis_fname)[1].header
antennas = h['ANTENNAS'].split('-')
iiter = 0
nants=10
for i in range(nants-1):
        for j in range(i+1,nants):
                a1=int(antennas[i]); a2=int(antennas[j]);
                print a1,a2,np.angle(np.nanmean(vis_ref[:,iiter]))*180./np.pi
                iiter += 1
                



# define grid of trials
# do chi squared test
da = 0.25*np.pi/3600./180. # rad
ngrid = 200
ogrid = np.zeros((ngrid,ngrid)) # dec, ra (ra does not need a fliplr)
for iRA in np.arange(ngrid):
        for iDEC in np.arange(ngrid):

                ra_off = (ngrid/2.-iRA-1)*da
                dec_off = (-ngrid/2.+iDEC+1)*da

                w_term = np.exp(-1j*2.*np.pi*(u*ra_off+v*dec_off)).astype(np.complex64)
                vis_test = vis_ref*w_term
                rmn = np.nanmean(vis_test.real)
                
                stat = np.nansum((vis_test.imag)**2.)+np.nansum((vis_test.real-rmn)**2.)
                stat /= sig2
                
                ogrid[iDEC,iRA] = stat

ogrid -= np.min(ogrid)
np.savez('/home/user/tmp/ogrid.npz',ogrid=ogrid)


# cumulate to provide confidence levels
arr = (np.sort(np.ravel(ogrid)))
relprob = np.exp(-arr.copy()/2.)
relprob /= np.sum(relprob)
cumprob = np.cumsum(relprob)
w68 = np.where(np.abs(cumprob-0.68)==np.min(np.abs(cumprob-0.68)))
w95 = np.where(np.abs(cumprob-0.95)==np.min(np.abs(cumprob-0.95)))
w99 = np.where(np.abs(cumprob-0.99)==np.min(np.abs(cumprob-0.99)))
print 'Conf levels (0.68, 0.95, 0.99):',arr[w68],arr[w95],arr[w99]


#ogrid = np.load('/home/user/tmp/ogrid.npz')['ogrid']

# Write results into FITS files
hdr = pf.Header([('SIMPLE',True),('BITPIX',-32),('ORIGIN','DSA-10 CLIP'), ('AUTHOR','H.K.VEDANTHAM'), ('BTYPE','LOGLIK'), ('BUNIT','LOGLIK'), ('OBJECT','LOC_SRC'), ('POL','I'), ('NAXIS',2), ('NAXIS1',ngrid), ('NAXIS2',ngrid), ('IMTYPE','DIRTY'), ('EPOCH',2000.0), ('EQUINOX',2000.0), ('MJD',toa), ('CTYPE1','RA---SIN'), ('CTYPE2','DEC--SIN'), ('CUNIT1','DEGREE'), ('CUNIT2','DEGREE'), ('CRVAL1',ra*180./np.pi), ('CRVAL2',dec*180./np.pi), ('CRPIX1',ngrid/2), ('CRPIX2',ngrid/2), ('CDELT1',-da*180./np.pi), ('CDELT2',da*180./np.pi)])

pf.writeto(odir+'/'+oname+'.fits',ogrid,hdr,clobber=True)





