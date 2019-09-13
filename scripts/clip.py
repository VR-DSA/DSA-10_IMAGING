#point_dec = (22.0+00.0/60.0+52.0/3600.0)*np.pi/180.0 CLIP libraries: CaLibration and Imaging Pipeline
# Author: Harish Vedantham, California Institute of Technology
# March 2017
#
from const import *
import pyfits as pf
import matplotlib
#matplotlib.use('Agg')
import numpy as np
from numpy.linalg import pinv
from numpy.random import randn as rand
import matplotlib.pyplot as plt
import scipy
from scipy.special import j1 as j1
import astropy
from astropy import units as unt
from astropy.coordinates import ITRS, FK5, SkyCoord
from astropy.time import Time
import pwkit.environments.casa.util as casautil
qa = casautil.tools.quanta()
me = casautil.tools.measures()
from threading import Thread as Proc
CJ=np.conjugate
from numba import jit
#
#
#
def running_median(d,n=20):
        dout=np.zeros(d.shape,dtype=d.dtype)
        for i in range(len(d)):
                xmin=max(0,i-n/2); xmax=min(len(d),i+n/2)
                dout[i]=np.nanmedian(d[xmin:xmax])
        return dout

def medmad(x):
        # Return the median and mediam-absilute-deviation of array x (1D only; else flattened)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.absolute(x-med))
        return med,mad

def plot_dynspec(d,fname='dynspec.png',show=0):     # Plot the dynamic spectrum (assume real valued input) d is dims of time,freq
        d_freq = np.nanmedian(np.transpose(d),axis=0)
        d_time = np.nanmedian(np.transpose(d),axis=1)
        med,mad=medmad(d)

	cmap = matplotlib.cm.gray
	cmap.set_bad('b',1.)
        plt.figure(figsize=(12,12))
        ax1=plt.subplot2grid((5,5),(0,0),colspan=4,rowspan=4)
        #ax1.imshow(d,origin='lower',aspect='auto',vmin=med-1.5*mad,vmax=med+1.5*mad,cmap=cmap)
        ax1.imshow(d,origin='lower',aspect='auto',cmap=cmap)
        ax1.set_ylabel('time index'); ax1.set_xlabel('channel index')
        ax2=plt.subplot2grid((5,5),(4,0),rowspan=1,colspan=4)
        ax2.plot(d_time,'k')
        med,mad = medmad(d_time)
        #ax2.set_ylim([med-10*mad,med+10*mad])
        ax2.set_xlabel('channel index')
        ax2.set_ylabel('median flux [Jy]')
	ax2.set_xlim([0,len(d_time)])
        ax3=plt.subplot2grid((5,5),(0,4),rowspan=4,colspan=1)
        ax3.plot(d_freq,range(len(d_freq)),'k')
        med,mad=medmad(d_freq)
        #ax3.set_xlim([med-10.0*mad,med+20*mad])
	ax3.set_ylim([0,len(d_freq)])
        ax3.set_xlabel('median flux [Jy]')
        ax3.set_ylabel('time index')
        plt.tight_layout()
        if show:
                plt.show(); plt.close()
        plt.savefig(fname,bbox_inches='tight')
	plt.close()
	return 
#
#
def acplot(fname,ifplot=0,pltfname="acplot.pdf",txtfname="acplot.txt"):
	h=pf.open(fname)[1].header
	nt=h['NAXIS2']/NF
	nt_avg = int(np.round(60./TSAMP))
	#nt_avg = 1
	n=nt*NF
        mjd0=h['MJD']                                                         # Start MHD is in header 
        mjd1=mjd0+TSAMP*float(nt)/SinD
        st0=Time(mjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian
        st = np.angle(np.exp(1j*( st0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );

        mjd=np.linspace(mjd0,mjd1,nt+1)[:nt]

	d1=np.reshape(pf.open(fname)[1].data['A11'][:n],(nt,NF))[:,800:900].astype('f')
	d2=np.reshape(pf.open(fname)[1].data['A22'][:n],(nt,NF))[:,800:900].astype('f')
	d3=np.reshape(pf.open(fname)[1].data['A33'][:n],(nt,NF))[:,800:900].astype('f')
	d4=np.reshape(pf.open(fname)[1].data['A44'][:n],(nt,NF))[:,800:900].astype('f')

	n1=np.std(np.diff(d1,axis=1))/np.sqrt(2.)
	n2=np.std(np.diff(d2,axis=1))/np.sqrt(2.)
	n3=np.std(np.diff(d3,axis=1))/np.sqrt(2.)
	n4=np.std(np.diff(d4,axis=1))/np.sqrt(2.)

	tp1 = np.zeros(nt/nt_avg,dtype='f')
	tp2 = np.zeros(nt/nt_avg,dtype='f')
	tp3 = np.zeros(nt/nt_avg,dtype='f')
	tp4 = np.zeros(nt/nt_avg,dtype='f')
	st_avg = np.zeros(nt/nt_avg,dtype='f')
	for i in range(nt/nt_avg):
		tp1[i]=np.mean(d1[i*nt_avg:(i+1)*nt_avg,:])/n1
		tp2[i]=np.mean(d2[i*nt_avg:(i+1)*nt_avg,:])/n2
		tp3[i]=np.mean(d3[i*nt_avg:(i+1)*nt_avg,:])/n3
		tp4[i]=np.mean(d4[i*nt_avg:(i+1)*nt_avg,:])/n4
		st_avg[i]=Time(np.mean(mjd[i*nt_avg:(i+1)*nt_avg]),format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian*12./PI
	f=open(txtfname,'w')
	for i in range(len(st_avg)):
		f.write("%f, %f, %f, %f, %f\n"%(st_avg[i],tp1[i],tp2[i],tp3[i],tp4[i]))
	f.close()
	if ifplot:
		tp1/=np.median(tp1)/SEFD;
		tp2/=np.median(tp2)/SEFD;
		tp3/=np.median(tp3)/SEFD;
		tp4/=np.mean(tp4)/SEFD;
		plt.plot(st_avg,tp1-SEFD,label='A11')
		plt.plot(st_avg,tp2-SEFD,label='A22')
		plt.plot(st_avg,tp3-SEFD,label='A33')
		plt.plot(st_avg,tp4-SEFD,label='A44')
		plt.ylim([-50,150])
		plt.xlabel('Sidereal time [hr]'); plt.ylabel('Normalized AC [linear units]')
		plt.legend(['A11','A22','A33','A44'],frameon=False);
		plt.tight_layout()
		plt.savefig(pltfname)
		plt.close()
		return
	else:
		return (st_avg,tp1,tp2,tp3,tp4)

def ccplot(fname, cc_code='B13'):
        h=pf.open(fname)[1].header
        nt=h['NAXIS2']/NF
        nt_avg = int(np.round(60./TSAMP))
        #nt_avg = 1
        n=nt*NF
        mjd0=h['MJD']                                                         # Start MHD is in header 
        mjd1=mjd0+TSAMP*float(nt)/SinD
        st0=Time(mjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian
        st = np.angle(np.exp(1j*( st0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );

        mjd=np.linspace(mjd0,mjd1,nt+1)[:nt]

        d1=(np.reshape(pf.open(fname)[1].data[cc_code+'R'][:n],(nt,NF))[:,800:810]+1j*np.reshape(pf.open(fname)[1].data[cc_code+'I'][:n],(nt,NF))[:,800:810]).astype(np.complex64)
	noise_r = np.std(np.diff(d1.real,axis=0))/(2.**0.5)
	noise_i = np.std(np.diff(d1.imag,axis=0))/(2.**0.5)
	noise = (noise_r+noise_i)/2.
	print "Noise = %f+1j%f"%(noise_r,noise_i)
	sig = np.mean(np.absolute(d1),axis=1)

	st_avg = np.zeros(nt/nt_avg,dtype='f')
	snr_avg = np.zeros(nt/nt_avg,dtype='f')
	for i in range(nt/nt_avg):
		snr_avg[i]=np.mean(sig[i*nt_avg:(i+1)*nt_avg])/noise
		st_avg[i]=Time(np.mean(mjd[i*nt_avg:(i+1)*nt_avg]),format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian*12./PI

	return (st_avg,snr_avg)	
	
def fitsinfo(fname,quiet=True):
	h = pf.open(fname)[1].header
	mjd0=h['MJD']+MOFF                                                         # Start MHD is in header 
	N = h['NAXIS2']/NF
	mjd1=mjd0+h['TSAMP']*float(N)/SinD
        st0=Time(mjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian
        st1=Time(mjd1,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian
	if not quiet:
		print "\n-------------------FITSINFO-------------------"
		print "File name: %s"%fname
		print "Start MJD: %f   Sidereal time: %f hrs"%(mjd0, st0*12./PI)
		print "Sampling interval : %f sec   Num samples = %d"%(h['TSAMP'], N)
		print "End MJD: %f    Sidereal time: %f"%(mjd1, st1*12./PI)
		print "Antennas: %s"%h['ANTENNAS']
		print "------------------------------------------------" 
	return (mjd0,mjd1,st0,st1)

def parse_NVSS(fname):
	# For now only stokes I
	# RA DEC FLUX (mJy) EXTENT_X EXTENt_Y POS_ANGLE --> format of imput ASCII file
	# 00 04 57.12 +12 48 18.9  1069.5  86.7  20.4 -81.1  ---> Example line
	#
	sky_model=[]	
	with open(fname) as f:
		for line in f:
			w=line.split()
			name="J"+w[0]+w[1]+w[3]+w[4]
			ra = np.pi/12.0*(float(w[0])+float(w[1])/60.0+float(w[2])/3600.0)
			dec = np.pi/180.0*(np.absolute(float(w[3]))+float(w[4])/60.+float(w[5])/3600.)
			dec*=np.sign(float(w[3]))
			flux=float(w[6])
			if w[7][0]=='<' or w[8][0]=='<':
				bmin=0; bmaj=0; pos=0.0
			else:
				bmaj=float(w[7])
				bmin=float(w[8])
				pos=float(w[9])
			sky_model.append([ra,dec,flux,bmaj,bmin,pos,name])
	return sky_model
#
#
def mjd2ST(t_mjd,lon_deg=-118.283400,lat_deg=37.233386):	# Default lat,lon <==> centre of MMA T
	t=Time(t_mjd,format='mjd',location=(lon_deg*unt.deg,lat_deg*unt.deg))
	return t.sidereal_time('apparent').radian
#
def get_infield_srcs(sky_model,t_mjd, pointing=POINTING,lon_deg=LON, lat_deg=LAT, dish_dia=4.5,freq=1.5e9,minflux=2e3,show=0):
# sky_model is the output of parse_NVSS. Each entry in the list has format of [ra,dec,flux,bmaj,bmin]
# pointing is the dec of pointing in radian
# dish_dia and freq are used to comput the primary beam (Airy func)
# Only reurn sources with apparent flux > 10.0 Jy
	dec=pointing
	loc=(lon_deg*unt.deg, lat_deg*unt.deg)
	lst=mjd2ST(t_mjd)

	lam=C/freq; bw=lam/dish_dia
	infield_model = []
	for s in sky_model:
		dis=(  (dec-s[1])**2.0 + (np.cos(0.5*(dec+s[1]))*(lst-s[0]))**2.0  )**0.5
		if np.amin(dis)<bw:	
			arg = np.pi*np.sin(dis)/bw
			gain= (2.0*j1(arg)/arg)**2.0				
			pt=SkyCoord(ra=s[0],dec=s[1],unit='rad',frame=FK5,equinox='J2000')
			ef=pt.transform_to(ITRS)
			uvec=[ef.x.value,ef.y.value,ef.z.value]
			if s[2]*gain > minflux:
				infield_model.append([s[0],s[1],s[2]*gain,s[3],s[4],s[5],s[6]])
	if show:
		plt.figure(figsize=(8,8))
		for s in infield_model:
			plt.plot(lst-s[1],dec-s[2],'ko')
			plt.text(lst-s[1],dec-s[2],'%.1f'%s[3])
			print "%.1f', %.1f', %.1f Jy"%((lst-s[1])*180.0/np.pi*60.,(dec-s[2])*180.0/np.pi*60.,s[3]/1e3)
		plt.title('HA: %s, DEC: %s'%(ra2str(lst),dec2str(dec)))
		plt.xlabel('RA offset [radian]'); plt.ylabel('DEC offset [radian]')
		plt.xlim([-bw*1.2,bw*1.2]); plt.ylim([-bw*1.2,bw*1.2])
		plt.tight_layout()
		plt.savefig('infield.pdf')
		plt.close()
	return infield_model,lst
#
#
def ra2str(ra):
	ra_deg = ra*12.0/np.pi
	ra_h = np.floor(ra_deg)
	ra_m = np.floor(60.0*(ra_deg-ra_h))
	ra_s = 3600.0*(ra_deg-ra_h-ra_m/60.0)
	return "%2d:%2d:%.2f"%(ra_h,ra_m,ra_s)
#
def dec2str(dec):
	dec_deg = np.absolute(dec*180.0/np.pi)
	dec_sign = np.sign(dec)
	dec_d = np.floor(dec_deg)
	dec_m = np.floor(60.0*(dec_deg-dec_d))
	dec_s = 3600.0*(dec_deg-dec_d-dec_m/60.0)
	return "%+d:%2d:%.2f"%(dec_sign*dec_d,dec_m,dec_s)

def l2norm(d):
	return (np.sum(  (np.array(d))**2.0   ))**0.5
#
#
#
def calc_uvw(bl_itrf,t_mjd,pointing=POINTING,src=None,noepoch=False):
# bl_itrf must be a 3 element vector  with baseline in ITRF
# pointing must be a the pointing declination in radian
# t_mjd must be a 1D numpy array of mjd values at which uvw is to be computed
# if src is None (default) compute uvw of the pointing centre else do so for src direction
# the latter is useful for predicting visibilities, the former for imaging
# Dont call this func directly unless you have to. Use the feeder function fast_aalc_uvw
#
        nt=len(t_mjd)		# num of baselines and time-steps
	st=mjd2ST(t_mjd)				# RA of pointing at each epoch

        bls = me.baseline('itrf', qa.quantity(bl_itrf[0], 'm'), qa.quantity(bl_itrf[1], 'm'), qa.quantity(bl_itrf[2], 'm'))
        #me.doframe(me.observatory('DSA-10')) 		# Observatory (will look for entry in casacore measures:directory (see ~/.casarc)
        me.doframe(me.observatory('OVRO_MMA')) 		# Observatory (will look for entry in casacore measures:directory (see ~/.casarc)

        u = np.zeros( (nt) )
        v = np.zeros( (nt) )
        w = np.zeros( (nt) )

	if src==None:

		for i in range(nt):
			#me.doframe(me.direction('J2000', qa.quantity(st[i],'rad'),qa.quantity(pointing,'rad')))	
			#me.doframe(me.direction('AZEL', qa.quantity(np.pi,'rad'),qa.quantity(75.0*np.pi/180.0,'rad')))
                        me.doframe(me.direction('HADEC', qa.quantity(0.0,'rad'),qa.quantity(pointing,'rad')))
                        
			#me.doframe(me.epoch('utc',qa.quantity(t_mjd[i],'d')))
			me.doframe(me.epoch('utc',qa.quantity(REF_EPOCH,'d')))
			uvw=me.touvw(bls)[1]['value']
			u[i],v[i],w[i]=uvw[0],uvw[1],uvw[2]
	else:

		me.doframe(me.direction('J2000', qa.quantity(src[0],'rad'),qa.quantity(src[1],'rad')))
		for i in range(nt):
			me.doframe(me.epoch('utc',qa.quantity(t_mjd[i],'d')))
			uvw=me.touvw(bls)[1]['value']
			u[i],v[i],w[i]=uvw[0],uvw[1],uvw[2]
	return u,v,w
#
#
def fast_calc_uvw(bl_itrf,t_mjd,pointing=POINTING,max_err=0.2,src=None,noepoch=False):
# feeder function to calc_uvw. Calls calc_uvw with a sparse t_mjd grid and then interpolates to save time
# max_err is the maximum linear interpolation error we can tolerate in meter. Used to decide resolution of coarse grid
# max_err should typically be << dish diameter. Default is 0.5 for the 4.5m DSA dishes.
#
	if len(t_mjd)==1:
		ug,vg,wg=calc_uvw(bl_itrf,t_mjd,pointing,src,noepoch=noepoch)
		return ug,vg,wg

	th_max = np.arccos(1.0-max_err/l2norm(bl_itrf))	# Max baseline rotation that can be tolerated
	dt_max = th_max/PI2*SinD
	dt = SinD*(t_mjd[1]-t_mjd[0])
	if dt>dt_max:
		u,v,w=calc_uvw(bl_itrf,t_mjd,pointing,src,noeopch=noepoch)
		return u,v,w
	else:
		dt=dt_max/SinD
		t_grid = np.arange(t_mjd[0]-dt,t_mjd[-1]+dt,dt)
		ug,vg,wg=calc_uvw(bl_itrf,t_grid,pointing,src,noepoch=noepoch)
		return np.interp(t_mjd,t_grid,ug), np.interp(t_mjd,t_grid,vg),np.interp(t_mjd,t_grid,wg)

def ang_dist(src,lst,pointing=POINTING):
# Compute angle distance from pointing centre of a src at a given lst
# src is (ra,dec)
# pointing is pointing of declination
# lst is the local sidereal time in radian
# This calculation is approximate!!!
	avg_dec = 0.5*(src[1]+pointing)
	dra = lst-src[0]
	ddec = pointing-src[1]
	return ((ddec)**2.0 + (dra *np.cos(avg_dec))**2.0)**0.5
#
def pb_resp(dis,freq,dish_dia=4.5):
# Compute the primary beam gain for a given angular offset from pointing
# If things are slow then we could do a look-up table for this
# dis a 1D array of angular offsets (radian)
# freq is 1D array of frequenies in Hz
	lam=299792458.0/freq
	Dbyl = dish_dia/lam
	pb = [ [ (2.0*j1(np.pi*dis[i]*Dbyl[j])/(np.pi*dis[i]*Dbyl[j]))**2.0 for j in range(len(freq))] for i in range(len(dis))]
	return np.array(pb).astype(np.float32)	# Dims of pb are n_dis x n_freq
# 
def predict_vis(bls_itrf,t_mjd,fqs,src_model,pointing,quiet=True):
#
	if not quiet:
		print "\n-----------------------PREDICT--------------------"
	nt = len(t_mjd)
	nf=len(fqs)
	lst=mjd2ST(t_mjd)
	spind_fun=np.tile(((fqs/1.4e9)**-0.7 ).astype(np.float32),(nt,1))
	pb=[]
	for src in src_model:
		dis = ang_dist([src[0],src[1]],lst)
		pb.append(pb_resp(dis,fqs))

	model = [predict_vis_worker(bls_itrf[i],t_mjd,src_model,spind_fun,pb,quiet,fqs) for i in range(len(bls_itrf))];
	if not quiet:
		print "--------------------------------------------------"
	return model
'''
	for bl_itrf in bls_itrf:
		vis = np.zeros((nt,nf),dtype=np.complex64)
		up,vp,wp=fast_calc_uvw(bl_itrf,[t_mjd[nt/2]])	# Since we are not tracking the u,v,w of the pointing
							# centre only changes by about 2mm/day 
		print "Predict: bl="+str(bl_itrf)+"src = "+str(src_model[:3])+" UVW="+str([up[0],vp[0],wp[0]])
		u,v,w=fast_calc_uvw(bl_itrf,t_mjd,src=[src_model[0],src_model[1]])
		vis+=src_model[2]*spind_fun*np.exp(arg*np.transpose(np.tile((w-wp),(nf,1)))).astype(np.complex64)
		model.append(vis)
	return model
'''

def predict_vis_worker(bl_itrf,t_mjd,src_model,spind_fun,pb,quiet,fqs):
	nt=len(t_mjd); nf=len(fqs)
	exp_arg=np.tile(1j*PI2*fqs/C,(nt,1))
	lam2_inv=np.tile((fqs/C)**2.0,(nt,1))
	up,vp,wp=fast_calc_uvw(bl_itrf,[t_mjd[nt/2]])   # Since we are not tracking the u,v,w of the pointing
        #up,vp,wp=fast_calc_uvw(bl_itrf,[57900.])   # Since we are not tracking the u,v,w of the pointing
                                                        # centre only changes by about 2mm/day 
	vis=np.zeros((nt,nf),dtype=np.complex64);
	i_src=0
	for src in src_model:
		if not quiet:
	        	print "Predict: bl="+str(bl_itrf)+"src = "+str(src[:3])+" UVW="+str([up[0],vp[0],wp[0]])
	        u,v,w=fast_calc_uvw(bl_itrf,t_mjd,src=[src[0],src[1]])
		
		if src[3]<10.0:
	        	vis+= src[2]*spind_fun*np.exp(exp_arg*np.transpose(np.tile((w-wp),(nf,1)))).astype(np.complex64) * pb[i_src]

		else:
			#sigmaj2=2.0*(np.pi*src[3]*(np.pi/180./3600.0)/2.355)**2.0; sigmin2=2.0*(np.pi*src[4]*(np.pi/180.0/3600.0)/2.355)**2.0; 
			sigmaj2=( (np.pi*src[3]*(np.pi/180./3600.0))**2.0 )/(4.0*np.log(2.0)); 
			sigmin2=( (np.pi*src[4]*(np.pi/180./3600.0))**2.0)/(4.0*np.log(2.0)); 
			pos=src[5]*np.pi/180.0; spos=np.sin(pos); cpos=np.cos(pos)
			proj_maj = v*cpos+u*spos; proj_min = -v*spos+u*cpos
			res_vis=np.exp(-np.transpose(np.tile(sigmaj2*proj_maj**2.0+sigmin2*proj_min**2.0,(nf,1)))*lam2_inv).astype(np.float32)
                        vis+= res_vis*src[2]*spind_fun*np.exp(exp_arg*np.transpose(np.tile((w-wp),(nf,1)))).astype(np.complex64) #* pb[i_src]

		i_src+=1
		
	return vis
#	
#
def delay_fit(dat,bl,Imid,fqs,Nfft=4096*2,basename=None):
        nt,nf=dat[0].shape
	freq=np.flipud(fqs)
        df=freq[1]-freq[0]
	t_vec=np.linspace(-1.0/2./df,1.0/2./df,Nfft+1)[:Nfft]
	dt=t_vec[1]-t_vec[0]
	delays=[]
	sigs=[]
	vis=np.zeros(dat[0].shape,dtype=dat[0].dtype)
	if basename!=None:
		plt.figure(figsize=(9,9))
                plt.tick_params(axis='both',which='major',labelsize=4)
                plt.tight_layout(pad=1)
	for i in range(len(dat)):
	        vis[:,:]=np.fliplr(dat[i])
		vis_vec= np.nanmean(vis[max(0,Imid-500):min(nt,Imid+500),:],axis=0) 
		vis_vec[np.where(np.isnan(vis_vec.real))]=0.0+1j*0.0
        	ds=np.absolute(np.fft.fftshift(np.fft.fft(vis_vec,n=Nfft)/2048.));
		maxarg = np.argmax(ds)
                xmin = 0
                xmax = Nfft
		#xmin = max(0,maxarg-300); xmax = min(Nfft,maxarg+300)
		med=np.median(ds[xmin:xmax]); mad = np.median(np.absolute(ds[xmin:xmax]-med));
		significance = (ds[maxarg]-med)/mad/1.5
                if significance<8.:
                        xmin = 0
                        xmax=Nfft
		sigs.append(significance)
		delays.append(float(maxarg-Nfft/2)*dt)
		if basename!=None:
			ax=plt.subplot(9,5,i+1)
       			ax.plot(t_vec[xmin:xmax]*1e6,ds[xmin:xmax],label="Sig=%.1f"%significance); 
			ax.plot([t_vec[xmin]*1e6, t_vec[xmax-1]*1e6],[med-1.5*mad,med-1.5*mad],'-',color='0.7',linewidth=1.5)
			ax.plot([t_vec[xmin]*1e6, t_vec[xmax-1]*1e6],[med+1.5*mad,med+1.5*mad],'-',color='0.7',linewidth=1.5)
			ax.legend(frameon=False)
			ax.set_yticklabels([])
			#ax.set_xlabel(r'Delay [$\mu s$]',fontsize=8)
			ax.set_ylabel('%d-%d (%.1f ns)'%(bl[i][0]+1,bl[i][1]+1,delays[-1]*1e9),fontsize=8)

	if basename!=None:
       		plt.suptitle(basename);plt.savefig('%s.png'%basename); plt.close() 	
	return delays, sigs

#
# extracts a specific number of samples
def extract_samples(fname,stmid,seg_len,pol):

        hdr=pf.open(fname)[1].header		# FITS header
        dat=pf.open(fname)[1].data		# FITS data
        n=hdr['NAXIS2']				# Number of rows in data
        nt=n/NF					# Number of integrations = nt
        n=nt*NF					# Cut-off (if) incomplete last integration

        mjd0=hdr['MJD']+MOFF+TIME_OFFSET/SinD        # Start MHD is in header 
        mjd1=mjd0+TSAMP*float(nt)/SinD + TIME_OFFSET/SinD	# End MJD (TSAMP is defined in const.py)
	if (mjd1-mjd0)>=1:			
		print "Data covers > 1 sidreal day. Only the first segment will be extracted"
        st0=Time(mjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian	# sStarting sidereal time 

	
        mjd= (mjd0+np.arange(0.0,float(nt),1.0)*TSAMP/SinD)					# MJD vector
	st = np.angle(np.exp(1j*( st0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );	# Sidereal time vector (SinSD is # secs in a sidereal day)
	
        #I1=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*stmid)*np.exp(-1j*seg_len)));        
        #I2=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*stmid)*np.exp(1j*seg_len)));
        # midpoint sample
        I0=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*(stmid))));       
        iI1 = I0-int(seg_len/2)
        iI2 = I0+int(seg_len/2)
        I1 = I0-int(seg_len/2)
        I2 = I0+int(seg_len/2)        
        a1=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a2=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a3=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a4=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a5=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a6=np.zeros((I2-I1,NF1),dtype=np.complex64)
        omjd=np.zeros(I2-I1)
        ost=np.zeros(I2-I1)

        print "Extract: %d ----> %d sample; transit at %d"%(I1,I2,I0)
	print "----------------------------------------------"
        
        # check whether other files need to be opened
        if I1<0:

                # check for previous file
                pfln = int(fname.split('.fits')[0].split('_')[2])-1
                pfname = fname.split('.fits')[0].split('_')[0]+'_'+fname.split('.fits')[0].split('_')[1]+'_'+str(pfln)+'.fits'
                try:
                        pdat=pf.open(pfname)[1].data
                        phdr=pf.open(pfname)[1].header
                        J1=0
                        J2=-iI1
                        I1=nt+iI1
                        I2=nt
                        p1=Proc(target=extract_samples_worker,args=(a1,pdat,'%s12'%pol,nt,I1,I2,J1,J2)); p1.start()
                        p2=Proc(target=extract_samples_worker,args=(a2,pdat,'%s13'%pol,nt,I1,I2,J1,J2)); p2.start()
                        p3=Proc(target=extract_samples_worker,args=(a3,pdat,'%s14'%pol,nt,I1,I2,J1,J2)); p3.start()
                        p4=Proc(target=extract_samples_worker,args=(a4,pdat,'%s23'%pol,nt,I1,I2,J1,J2)); p4.start()
                        p5=Proc(target=extract_samples_worker,args=(a5,pdat,'%s24'%pol,nt,I1,I2,J1,J2)); p5.start()
                        p6=Proc(target=extract_samples_worker,args=(a6,pdat,'%s34'%pol,nt,I1,I2,J1,J2)); p6.start()
                        p1.join(); p2.join(); p3.join(); p4.join(); p5.join(); p6.join()
                        
                        pmjd0=phdr['MJD']+MOFF+TIME_OFFSET/SinD        # Start MHD is in header 
                        pmjd1=pmjd0+TSAMP*float(nt)/SinD + TIME_OFFSET/SinD	# End MJD (TSAMP is defined in const.py)
	                pst0=Time(pmjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian	# sStarting sidereal time 
                        pmjd= (pmjd0+np.arange(0.0,float(nt),1.0)*TSAMP/SinD)					# MJD vector
	                pst = np.angle(np.exp(1j*( pst0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );	# Sidereal time vector (SinSD is # secs in
                        omjd[J1:J2]=pmjd[I1:I2]
                        ost[J1:J2]=pst[I1:I2]
                
                except:
                        print 'NO PREVIOUS FILE: cannot open',pfname

                J1=-iI1
                J2=nt
                I1=0
                I2=iI2
                p1=Proc(target=extract_samples_worker,args=(a1,dat,'%s12'%pol,nt,I1,I2,J1,J2)); p1.start()
                p2=Proc(target=extract_samples_worker,args=(a2,dat,'%s13'%pol,nt,I1,I2,J1,J2)); p2.start()
                p3=Proc(target=extract_samples_worker,args=(a3,dat,'%s14'%pol,nt,I1,I2,J1,J2)); p3.start()
                p4=Proc(target=extract_samples_worker,args=(a4,dat,'%s23'%pol,nt,I1,I2,J1,J2)); p4.start()
                p5=Proc(target=extract_samples_worker,args=(a5,dat,'%s24'%pol,nt,I1,I2,J1,J2)); p5.start()
                p6=Proc(target=extract_samples_worker,args=(a6,dat,'%s34'%pol,nt,I1,I2,J1,J2)); p6.start()
                p1.join(); p2.join(); p3.join(); p4.join(); p5.join(); p6.join()
                omjd[J1:J2]=mjd[I1:I2]
                ost[J1:J2]=st[I1:I2]

                print 'EXTRACTED USING PREVIOUS FILE'
                
        elif I2>nt:

                # check for future file
                pfln = int(fname.split('.fits')[0].split('_')[2])+1
                pfname = fname.split('.fits')[0].split('_')[0]+'_'+fname.split('.fits')[0].split('_')[1]+'_'+str(pfln)+'.fits'
                try:
                        pdat=pf.open(pfname)[1].data
                        phdr=pf.open(pfname)[1].header
                        J1=nt-iI1
                        J2=nt
                        I1=0
                        I2=iI2-nt
                        p1=Proc(target=extract_samples_worker,args=(a1,pdat,'%s12'%pol,nt,I1,I2,J1,J2)); p1.start()
                        p2=Proc(target=extract_samples_worker,args=(a2,pdat,'%s13'%pol,nt,I1,I2,J1,J2)); p2.start()
                        p3=Proc(target=extract_samples_worker,args=(a3,pdat,'%s14'%pol,nt,I1,I2,J1,J2)); p3.start()
                        p4=Proc(target=extract_samples_worker,args=(a4,pdat,'%s23'%pol,nt,I1,I2,J1,J2)); p4.start()
                        p5=Proc(target=extract_samples_worker,args=(a5,pdat,'%s24'%pol,nt,I1,I2,J1,J2)); p5.start()
                        p6=Proc(target=extract_samples_worker,args=(a6,pdat,'%s34'%pol,nt,I1,I2,J1,J2)); p6.start()
                        p1.join(); p2.join(); p3.join(); p4.join(); p5.join(); p6.join()
                        
                        pmjd0=phdr['MJD']+MOFF+TIME_OFFSET/SinD        # Start MHD is in header 
                        pmjd1=pmjd0+TSAMP*float(nt)/SinD + TIME_OFFSET/SinD	# End MJD (TSAMP is defined in const.py)
	                pst0=Time(pmjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian	# sStarting sidereal time 
                        pmjd= (pmjd0+np.arange(0.0,float(nt),1.0)*TSAMP/SinD)					# MJD vector
	                pst = np.angle(np.exp(1j*( pst0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );	# Sidereal time vector (SinSD is # secs in
                        omjd[J1:J2]=pmjd[I1:I2]
                        ost[J1:J2]=pst[I1:I2]
                
                except:
                        print 'NO FUTURE FILE: cannot open',pfname

                J1=0
                J2=nt-iI1
                I1=iI1
                I2=nt
                p1=Proc(target=extract_samples_worker,args=(a1,dat,'%s12'%pol,nt,I1,I2,J1,J2)); p1.start()
                p2=Proc(target=extract_samples_worker,args=(a2,dat,'%s13'%pol,nt,I1,I2,J1,J2)); p2.start()
                p3=Proc(target=extract_samples_worker,args=(a3,dat,'%s14'%pol,nt,I1,I2,J1,J2)); p3.start()
                p4=Proc(target=extract_samples_worker,args=(a4,dat,'%s23'%pol,nt,I1,I2,J1,J2)); p4.start()
                p5=Proc(target=extract_samples_worker,args=(a5,dat,'%s24'%pol,nt,I1,I2,J1,J2)); p5.start()
                p6=Proc(target=extract_samples_worker,args=(a6,dat,'%s34'%pol,nt,I1,I2,J1,J2)); p6.start()
                p1.join(); p2.join(); p3.join(); p4.join(); p5.join(); p6.join()
                omjd[J1:J2]=mjd[I1:I2]
                ost[J1:J2]=st[I1:I2]

                print 'EXTRACTED USING FUTURE FILE'

        else:

                J1=0
                J2=nt
                I1=iI1
                I2=iI2
                p1=Proc(target=extract_samples_worker,args=(a1,dat,'%s12'%pol,nt,I1,I2,J1,J2)); p1.start()
                p2=Proc(target=extract_samples_worker,args=(a2,dat,'%s13'%pol,nt,I1,I2,J1,J2)); p2.start()
                p3=Proc(target=extract_samples_worker,args=(a3,dat,'%s14'%pol,nt,I1,I2,J1,J2)); p3.start()
                p4=Proc(target=extract_samples_worker,args=(a4,dat,'%s23'%pol,nt,I1,I2,J1,J2)); p4.start()
                p5=Proc(target=extract_samples_worker,args=(a5,dat,'%s24'%pol,nt,I1,I2,J1,J2)); p5.start()
                p6=Proc(target=extract_samples_worker,args=(a6,dat,'%s34'%pol,nt,I1,I2,J1,J2)); p6.start()
                p1.join(); p2.join(); p3.join(); p4.join(); p5.join(); p6.join()
                omjd[J1:J2]=mjd[I1:I2]
                ost[J1:J2]=st[I1:I2]


	return [a1,a2,a3,a4,a5,a6],ost,omjd,I0-iI1

        
        
#
def extract_segment(fname,stmid,seg_len,pol,quiet=True):
#
        hdr=pf.open(fname,ignore_missing_end=True)[1].header		# FITS header
        dat=pf.open(fname,ignore_missing_end=True)[1].data['VIS']	# FITS data
        nt=hdr['NAXIS2']			# Number of rows in data
        nchan=hdr['NCHAN'] # number of channels

        mjd0=hdr['MJD']+MOFF+TIME_OFFSET/SinD        # Start MHD is in header 
        mjd1=mjd0+TSAMP*float(nt)/SinD	# End MJD (TSAMP is defined in const.py)
	if (mjd1-mjd0)>=1:			
		print "Data covers > 1 sidreal day. Only the first segment will be extracted"
        st0=Time(mjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian	# sStarting sidereal time 

	
        mjd= (mjd0+np.arange(0.0,float(nt),1.0)*TSAMP/SinD)					# MJD vector
	st = np.angle(np.exp(1j*( st0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );	# Sidereal time vector (SinSD is # secs in a sidereal day)

	if not quiet:
		print "\n-------------EXTRACT DATA--------------------"
	        print "%d Time samples in data"%nt
	        print "LST range: %.1f --- (%.1f-%.1f) --- %.1f deg"%(st[0]*180./PI,(stmid-seg_len)*180./PI, (stmid+seg_len)*180./PI,st[-1]*180./PI)

	
        I1=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*stmid)*np.exp(-1j*seg_len)));        
        I2=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*stmid)*np.exp(1j*seg_len)));       
        I0=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*(stmid))));       


        mjd=mjd[I1:I2]
        st=st[I1:I2]
        dat = dat.reshape((nt,55,nchan,2,2))[I1:I2,:,:,:,:]

	if not quiet:
	        print "Extract: %d ----> %d sample; transit at %d"%(I1,I2,I0)
		print "----------------------------------------------"

        
        odata = []
        basels = [1,3,4,6,7,8,10,11,12,13,15,16,17,18,19,21,22,23,24,25,26,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53]
        if pol=='A':
                opol = 0
        if pol=='B':
                opol = 1
        for bae in basels:
                odata.append(dat[:,bae,:,opol,0]+1j*dat[:,bae,:,opol,1])

	return odata,st,mjd,I0-I1

def extract_segment_worker(a,dat,name,nt,I1,I2):
	a[:,:]=np.reshape(dat['%sR'%name][I1*NF:I2*NF],(I2-I1,NF))[:,F_START:F_END].astype('f')+1j*np.reshape(dat['%sI'%name][I1*NF:I2*NF],(I2-I1,NF))[:,F_START:F_END].astype('f')

def extract_samples_worker(a,dat,name,nt,I1,I2,J1,J2):
	a[J1:J2,:]=np.reshape(dat['%sR'%name][I1*NF:I2*NF],(I2-I1,NF))[:,F_START:F_END].astype('f')+1j*np.reshape(dat['%sI'%name][I1*NF:I2*NF],(I2-I1,NF))[:,F_START:F_END].astype('f')

        
def extract_ac_worker(a,dat,name,nt,I1,I2):
	a[:,:]=np.reshape(dat['%s'%name][I1*NF:I2*NF],(I2-I1,NF))[:,F_START:F_END].astype('f')


def extract_ac(fname,stmid,seg_len,pol):
        hdr=pf.open(fname)[1].header            # FITS header
        dat=pf.open(fname)[1].data              # FITS data
        n=hdr['NAXIS2']                         # Number of rows in data
        nt=n/NF                                 # Number of integrations = nt
        n=nt*NF                                 # Cut-off (if) incomplete last integration

        mjd0=hdr['MJD']+MOFF                         # Start MHD is in header 
        mjd1=mjd0+TSAMP*float(nt)/SinD          # End MJD (TSAMP is defined in const.py)
        if (mjd1-mjd0)>=1:
                print "Data covers > 1 sidreal day. Only the first segment will be extracted"
        st0=Time(mjd0,format='mjd',location=(LON*unt.deg,LAT*unt.deg)).sidereal_time('apparent').radian # sStarting sidereal time 


        mjd= (mjd0+np.arange(0.0,float(nt),1.0)*TSAMP/SinD)                                     # MJD vector
        st = np.angle(np.exp(1j*( st0+np.arange(0.0,float(nt),1.0)*TSAMP/SinSD*PI2 ) ) );       # Sidereal time vector (SinSD is # secs in a sidereal day)

        I1=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*stmid)*np.exp(-1j*seg_len)));
        I2=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*stmid)*np.exp(1j*seg_len)));
        I0=np.argmax(np.absolute(np.exp(1j*st)+np.exp(1j*(stmid))));

        a1=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a2=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a3=np.zeros((I2-I1,NF1),dtype=np.complex64)
        a4=np.zeros((I2-I1,NF1),dtype=np.complex64)

        p1=Proc(target=extract_ac_worker,args=(a1,dat,'%s11'%pol,nt,I1,I2)); p1.start()
        p2=Proc(target=extract_ac_worker,args=(a2,dat,'%s22'%pol,nt,I1,I2)); p2.start()
        p3=Proc(target=extract_ac_worker,args=(a3,dat,'%s33'%pol,nt,I1,I2)); p3.start()
        p4=Proc(target=extract_ac_worker,args=(a4,dat,'%s44'%pol,nt,I1,I2)); p4.start()

        p1.join(); p2.join(); p3.join(); p4.join();

        return [a1,a2,a3,a4]

#
def noise_calc(vis,bst=50,bsf=50):
	nt,nf=vis.shape
	nbt=nt/bst
	nbf=nf/bsf
	noise=np.zeros((nbt,nbf),dtype=np.float32)
	for i in range(nbt):
		for j in range(nbf):
			noise[i,j]=np.nanstd(np.diff(vis[i*bst:i*bst+bst,j*bsf:j*bsf+bsf].real,axis=0))/(2.0**0.5)
	return noise
	
def block_avg(vis,bst=50,bsf=50):
	nt,nf=vis.shape
	nbt=nt/bst
	nbf=nf/bsf
	vis1=np.zeros((nbt,nbf),dtype=vis.dtype)

	for i in range(nbt):
		for j in range(nbf):
			vis1[i,j]=np.nanmean(vis[i*bst:min(i*bst+bst,nt),j*bsf:min(j*bsf+bsf,nf)])
	return vis1
#
#
#
def flag_2d(d,bst=100,bsf=100):
	I=np.where(d.real==0.0)
	d[I]*=np.nan
	nt,nf=d.shape
	da=np.absolute(d)

	nbt=nt/bst
	nbf=nf/bsf

	daf=np.nanmedian(da,axis=0)
	for i in range(nbf):
		tp=daf[i*bsf:min(nf,(i+1)*bsf)]
		med=np.nanmedian(tp)
		mad=np.nanmedian(np.absolute(tp-med))
		I=np.where(np.absolute(tp-med)>5.*1.5*mad)[0]
	
		for j in I:
			d[:,i*bsf+j]*=np.nan
	
	da=np.absolute(d)
	dat=np.nanmedian(da,axis=1)
	for i in range(nbt):
		tp=dat[i*bst:min(nt,(i+1)*bst)]
		med=np.nanmedian(tp)
		mad=np.nanmedian(np.absolute(tp-med))
		I=np.where(np.absolute(tp-med)>5.*1.5*mad)[0]

		for j in I:
			d[i*bst+j,:]*=np.nan

	da=np.absolute(d)
	for i in range(nbt):
		for j in range(nbf):
			tp=da[i*bst:min(nt,i*bst+bst),j*bsf:min(nf,j*bsf+bsf)]
			med=np.nanmedian(tp)
			mad=np.nanmedian(np.absolute(tp-med))
			I=np.where(np.absolute(tp-med)>5.0*1.5*mad)
			for ii in range(len(I[0])):
				d[i*bst+I[0][ii],j*bsf+I[1][ii]]*=np.nan
	return
#
#
#
def remove_delay(vis,d,fqs):
	ps=[]
	for i in range(len(vis)):
		p=Proc(target=remove_delay_worker,args=(vis[i],d[i],fqs)); p.start(); ps.append(p)
	for p in ps:
		p.join()
#		
#
def remove_delay_worker(vis,delay,fqs):
	nt,nf=vis.shape
	fac=np.exp(-1j*PI2*fqs*delay).astype(np.complex64)
	vis*=np.tile(fac,(nt,1))
#
#
#
#
@jit
def HL(theta,bls,nants):
        H=np.zeros((len(bls)*2,2*nants),dtype=np.float32)
        L=np.zeros((len(bls)*2),dtype=np.float32)
        irow=0
        for bl in bls:
                a,b=bl
                tp=(theta[2*a]+1j*theta[2*a+1]) * (theta[2*b]-1j*theta[2*b+1])
                cl=[a*2,a*2+1,b*2,b*2+1]
                H[irow,cl] = [theta[2*b],theta[2*b+1],theta[2*a],theta[2*a+1]]
                L[irow] = tp.real
                irow+=1
                H[irow,cl] = [-theta[2*b+1],theta[2*b],theta[2*a+1],-theta[2*a]]
                L[irow] = tp.imag
                irow+=1
        return H,L

@jit
def HLabs(theta,bls,nants):
        H=np.zeros((len(bls),nants),dtype=np.float32)
        L=np.zeros((len(bls)),dtype=np.float32)
        irow=0
        for bl in bls:
                a,b=bl
                tp=theta[a] * theta[b]
                cl=[a,b]
                H[irow,cl] = [theta[b],theta[a]]
                L[irow] = tp
                irow+=1
	return H,L
	
@jit
def LMsolve(dat,noise,bls,niter=50,DROP=0.5,BOOST=10.0,lam=20.0):
#       lam is the dampinf parameter
#       BOOST and DROP are the factors by which lam is updated if new solution is worse or better than previous in terms of SOS
#       bls has baslines tuples
#       dat is a complex vector with visibilities
#       nant is the number of antennas
#       niter is the max number of iterations

	nants = 1+np.amax(np.array(bls))
	
        if np.any(np.isnan(dat.real)):
                return np.ones((nants),dtype=np.complex64)


#	Detrmine approx gains to whiten noise matrix
        gabs = np.ones(nants,dtype=np.float32)
        H,mod=HLabs(gabs,bls,nants)
        for iiter in range(10):
                HTH=np.dot(np.transpose(H),H)
                cov=pinv(HTH)
                d_gabs = np.dot(cov,np.dot(np.transpose(H),noise-mod))	
                gabs+=d_gabs
        
#	Whiten the noise covarianvce matrix
#	factor back gabs in the end
	H,mod=HLabs(gabs,bls,nants)
	dat/=mod
	noise/=mod

        #theta = np.ones((nants*2),dtype=np.float32)*np.mean(np.absolute(dat))/2.0**0.5
        theta = np.mean(np.absolute(dat))/(2.0**0.5) *np.ones(2*nants) 
        d=[]
        n=[]

        for dats in dat:
                d.append(dats.real)
                d.append(dats.imag)
        for nois in noise:
                n.append(nois)
                n.append(nois)
        d=np.array(d).astype('f')
        n=np.array(n).astype('f')

        sos=1e16; iiter=0;

        H,mod=HL(theta,bls,nants)
        W=np.zeros((2*len(bls),2*len(bls)),dtype=np.float32)
        for i in range(len(bls)):
                W[2*i,2*i]=1.0/noise[i]**2.0
                W[2*i+1,2*i+1]=1.0/noise[i]**2.0

        while(iiter<niter):
                HTH=np.dot(np.dot(np.transpose(H),W),H)
                HTH+=lam*np.diag(np.diag(HTH))
                #HTH+=(lam)*np.eye(nants*2)
                cov=pinv(HTH)
		

                d_theta = np.dot(cov,np.dot(np.dot(np.transpose(H),W),d-mod))
                theta+=d_theta

                H,mod=HL(theta,bls,nants)
                sos_new=(np.mean(((d-mod)/n)**2.0))**0.5

                if sos_new<sos:
                        lam*=DROP 
			lam+=1e-3
                else:   
                        theta-=d_theta
                        if lam<1e4:
                                lam*=BOOST
                sos=sos_new

                iiter+=1
	if sos>3.0:
		return np.ones((nants),dtype=np.complex64)*np.nan
	else:   
		g=theta[0::2]+1j*theta[1::2]
		for i in range(len(theta)/2):
			g[i]=gabs[i]*(theta[2*i]+1j*theta[2*i+1])
		norm=np.conjugate(g[0])/np.absolute(g[0])
		for i in range(len(g)):
			g[i]*=norm
	return g 

