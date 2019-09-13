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




def calc_image_uvw(vis_fname="output.npz", image_centre = [0.,0.], savez=True):
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
                tp4,tp5,tp6=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING)
                tp1,tp2,tp3=fast_calc_uvw(blen[i,:],t_mjd,pointing=POINTING,src=[image_centre[0],image_centre[1]])
                for j in range(len(toa)):
                        u[j,:,i]=tp1[j]/lam
                        v[j,:,i]=tp2[j]/lam
                        w[j,:,i]=tp3[j]/lam
                        wcorr[j,:,i]=(tp3[j]-tp6[j])/lam
                        
        
	return u,v,w,wcorr




def applycal(cal_fname_x, cal_fname_y, vis_fname,doCal=True):

	tp=np.load(cal_fname_x)
	anames_x=list(tp['aname'])
	taus_x=tp['ant_delays']
	g_x = tp['gains']
	bsf_x=tp['freq_cadence']	
	norm(g_x) # Normalize the gains (take only phases)
        
	tp=np.load(cal_fname_y)
	anames_y=list(tp['aname'])
	taus_y=tp['ant_delays']
	g_y=tp['gains']
	bsf_y=tp['freq_cadence']
	norm(g_y)
	
	g_x_interp = interp_phase(g_x,bsf_x)
	g_y_interp = interp_phase(g_y,bsf_y)

        f=FREQ1
        h = np.load(vis_fname)
        antennas = str(h['ANTENNAS']).split('-')
        nants = len(antennas)
        nt = h['NAXIS2']
        
        bline_delay_x=np.zeros((nants*(nants-1)/2),dtype=np.float32)
        bline_delay_y=np.zeros((nants*(nants-1)/2),dtype=np.float32)
        phase_x=np.zeros((nants*(nants-1)/2,NF1),dtype=np.float32)
        phase_y=np.zeros((nants*(nants-1)/2,NF1),dtype=np.float32)

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


        # read in data
	d=np.load(vis_fname)['vis'].astype(np.float32)
	cc = np.reshape(d,newshape=(nt,NF1,180))

        # fill in vis
	vis = np.zeros((nt,NF1,nants*(nants-1)/2),dtype=np.complex64)

        iiter=0
        for i in range(nants-1):
                for j in range(i+1,nants):
                        a1=int(antennas[i]); a2=int(antennas[j]);

                        fac_x=np.exp(-1j*PI2*f*bline_delay_x[iiter]-1j*phase_x[iiter,:]).astype(np.complex64)
	                fac_y=np.exp(-1j*PI2*f*bline_delay_y[iiter]-1j*phase_y[iiter,:]).astype(np.complex64)
	                vis[:,:,iiter] =( (cc[:,:,4*iiter]+1j*cc[:,:,4*iiter+1])*fac_x  + (cc[:,:,4*iiter+2]+1j*cc[:,:,4*iiter+3])*fac_y  ) /2.0
                        if a1==8:
                                vis[:,:,iiter] = (cc[:,:,4*iiter]+1j*cc[:,:,4*iiter+1])*fac_x
                        if a2==8:
                                vis[:,:,iiter] = (cc[:,:,4*iiter]+1j*cc[:,:,4*iiter+1])*fac_x

                        iiter+=1
                        
	return vis 


############################################################################
#
#		MAIN BEGINS HERE
#
############################################################################

# arguments: input cal npz name, output dir, RA/DEC (rad)
vis_fname = sys.argv[1]
n = sys.argv[2]
oname = sys.argv[3]

cco = SkyCoord(sys.argv[4],sys.argv[5],unit=(u.hourangle,u.deg))
ra = cco.ra.radian
dec = cco.dec.radian

toa = np.mean(np.load(vis_fname)['MJD'])
cal_fname_x = "/home/user/tmp/TST/Acal_offset_"+n+".npz"
cal_fname_y = "/home/user/tmp/TST/Bcal_offset_"+n+".npz"
time = Time(toa,format='mjd')
#ra = time.sidereal_time('mean',-118.2834*u.deg).value*(np.pi/180.)*15.
#dec = 73.7523*np.pi/180. 


# Calculate the u,v,w and delta_delay for image centre rotation
u,v,w,wcorr = calc_image_uvw(vis_fname=vis_fname, image_centre=[ra,dec], savez=False)

doCal=True
# Read visibilities and apply calibration solutions to them
vis = applycal(cal_fname_x=cal_fname_x,cal_fname_y=cal_fname_y,vis_fname=vis_fname, doCal=doCal)

# Phase rotate calibrated visibilities to the image centre
w_term = np.exp(-1j*2.0*np.pi*wcorr).astype(np.complex64)	# Multiplying by w_term shifts the phase centre to image_centre
vis*=w_term


# save
np.savez('/home/user/tmp/TST/'+oname+'.npz',vis=vis,src=oname,time=toa,ra=ra,dec=dec,f=FREQ1)
