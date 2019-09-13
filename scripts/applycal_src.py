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

        lam=C/f
        for i in range(nants*(nants-1)/2):
                tp4,tp5,tp6=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING)
                tp1,tp2,tp3=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING,src=[image_centre[0],image_centre[1]])
                u[:,i]=tp1/lam
                v[:,i]=tp2/lam
                w[:,i]=tp3/lam
                wcorr[:,i]=(tp3-tp6)/lam

        
	return u,v,w,wcorr



def applycal(cal_fname_x, cal_fname_y, vis_fname, burst_loc, burst_nsamp):

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
        
        iiter=0
        for i in range(nants-1):
                for j in range(i+1,nants):
                        a1=int(antennas[i]); a2=int(antennas[j]);

                        fac_x=np.exp(-1j*PI2*f*bline_delay_x[iiter]-1j*phase_x[iiter,:]).astype(np.complex64)
	                fac_y=np.exp(-1j*PI2*f*bline_delay_y[iiter]-1j*phase_y[iiter,:]).astype(np.complex64)
	                vis[:,iiter] =( (cc[:,4*iiter]+1j*cc[:,4*iiter+1])*fac_x  + (cc[:,4*iiter+2]+1j*cc[:,4*iiter+3])*fac_y  ) /2.0
                        if a1==8:
                                vis[:,iiter] = (cc[:,4*iiter]+1j*cc[:,4*iiter+1])*fac_x
                        if a2==8:
                                vis[:,iiter] = (cc[:,4*iiter]+1j*cc[:,4*iiter+1])*fac_x

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

cal_fname_x = odir+"/Acal_offset.npz"
cal_fname_y = odir+"/Bcal_offset.npz"
object_name = "FRB-CAND"
opfname_prefix="frb"
toa = pf.open(vis_fname)[1].header['MJD']
time = Time(toa,format='mjd')
#ra = time.sidereal_time('mean',-118.2834*u.deg).value*(np.pi/180.)*15.
#dec = 73.6*np.pi/180.
image_centre = [ra, dec]
print ra, dec
print 'Outputting at location',burst_loc


# Calculate the u,v,w and delta_delay for image centre rotation
u,v,w,wcorr = calc_image_uvw(vis_fname=vis_fname, image_centre=image_centre, savez=True)

# Read visibilities and apply calibration solutions to them
vis = applycal(cal_fname_x=cal_fname_x,cal_fname_y=cal_fname_y,vis_fname=vis_fname, burst_loc=burst_loc,burst_nsamp=burst_nsamp)

# Phase rotate calibrated visibilities to the image centre
w_term = np.exp(-1j*2.0*np.pi*wcorr).astype(np.complex64)	# Multiplying by w_term shifts the phase centre to image_centre
vis*=w_term

np.savez(odir+'/'+oname+'.npz',vis=vis,src=oname,time=toa,ra=ra,dec=dec,f=FREQ2)
