import sys
from clip import extract_segment, predict_vis, delay_fit, remove_delay, block_avg, fitsinfo, noise_calc, LMsolve, flag_2d, plot_dynspec, extract_ac
from clip_bkeep import read_src_model, yield_all_inbetween, yield_cal_srcs
from const import *
import matplotlib.pyplot as plt, pylab
import pyfits as pf
from multiprocessing import Process as Proc1
import os



# Predefinitions
freq_cadence=5 # average to 25 chans per sub-band for calibration
quiet=True
bst=100		# Block size for time avg and flagging
bsf=100		# Block size for freq flagging
src = 'J2014+2334'
fl = sys.argv[1]
pol = sys.argv[2]

# fill antennas and baseline lengths
aname = pf.open(fl,ignore_missing_end=True)[1].header['ANTENNAS'].split('-')
tp=np.loadtxt('antpos_ITRF.txt')
tpe=np.loadtxt('antpos_ITRF_err.txt')
fqs=pf.open(fl,ignore_missing_end=True)[1].header['FCH1']*1e6-(np.arange(125)+0.5)*2.*2.5e8/2048.

blen=[]
blene=[]
bname=[]
for i in np.arange(9)+1:
        for j in np.arange(i):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                bname.append([a1,a2])
                blen.append(tp[a1,1:]-tp[a2,1:])
                blene.append(tpe[a1,1:]-tpe[a2,1:])
                

print "Antennas: "+str(aname)
print "Baseline: "+str(bname)
print "Source "+str(src)


src_model=read_src_model(src)
stmid=src_model[0][0]
data=[]

# READ DATA - extract +-1 deg from transit
data,st,mjd,Imid=extract_segment(fname=fl,stmid=stmid,seg_len=8.0/60.0/24.0*PI2, pol=pol, quiet=False)

nsamps = data[0].shape[0]
model=predict_vis(blen,mjd,fqs,src_model=src_model,pointing=POINTING,quiet=quiet)	# Predict the visibilities
modele=predict_vis(blene,mjd,fqs,src_model=src_model,pointing=POINTING,quiet=quiet)	# Predict the visibilities	
dbym=[]	# data by model
dbyme=[]	# data by model

f = block_avg(np.reshape(fqs,(1,len(fqs))),bst=1,bsf=freq_cadence).flatten()	

for i in range(len(data)):
	flag_2d(data[i],bst=bst,bsf=bsf)	
	dbym.append((data[i]/model[i])/np.abs(data[i]/model[i]))
        dbyme.append((data[i]/modele[i])/np.abs(data[i]/modele[i]))


# plot and fix delays
taus,sigs = delay_fit(dbym,bname,Imid,fqs,basename="fringe")
ant_delays = np.zeros(10)
refant = 7 # antenna 3
for i in range(len(aname)):
        if i==refant:
                ant_delays[i] = 0.0
        else:
                try:
                        ant_delays[i] = taus[bname.index([i,refant])]
                except:
                        ant_delays[i] = taus[bname.index([refant,i])]
        print "antenna delays = " + str(ant_delays)
b_delays = np.zeros((len(bname)))
for i in range(len(bname)):
        b_delays[i] = ant_delays[bname[i][1]]-ant_delays[bname[i][0]]
remove_delay(dbym,b_delays,fqs)
taus,sigs = delay_fit(dbyme,bname,Imid,fqs,basename="fringee")
ant_delays = np.zeros(10)
refant = 7 # antenna 3
for i in range(len(aname)):
        if i==refant:
                ant_delays[i] = 0.0
        else:
                try:
                        ant_delays[i] = taus[bname.index([i,refant])]
                except:
                        ant_delays[i] = taus[bname.index([refant,i])]
        print "antenna delays = " + str(ant_delays)
b_delays = np.zeros((len(bname)))
for i in range(len(bname)):
        b_delays[i] = ant_delays[bname[i][1]]-ant_delays[bname[i][0]]
remove_delay(dbyme,b_delays,fqs)

taus,sigs = delay_fit(dbym,bname,Imid,fqs,basename="fringe_fixed")
taus,sigs = delay_fit(dbyme,bname,Imid,fqs,basename="fringe_fixede")


#np.savez('play.npz',dbym=dbym,dbyme=dbyme)

#######

# now, divide out mean baseline phases from time-dependent ones, and plot mean phase v time on all baselines

time_avg = []
time_avge = []
dat_avg = []
dat_avge = []
for i in range(len(data)):
        time_avg.append(block_avg(dbym[i],bst=nsamps,bsf=freq_cadence))
        time_avg[i] = time_avg[i].flatten()
        time_avge.append(block_avg(dbyme[i],bst=nsamps,bsf=freq_cadence))
        time_avge[i] = time_avge[i].flatten()
        dat_avg.append(block_avg(dbym[i],bst=bst,bsf=freq_cadence))
        dat_avge.append(block_avg(dbyme[i],bst=bst,bsf=freq_cadence))
        dat_avg[i] /= np.abs(dat_avg[i])
        dat_avge[i] /= np.abs(dat_avge[i])
        time_avg[i] /= np.abs(time_avg[i])
        time_avge[i] /= np.abs(time_avge[i])

# divide out mean phases
nt = dat_avg[0].shape[0]
nf = dat_avg[0].shape[1]
for i in range(len(data)):
        for j in range(nt):
                dat_avg[i][j,:] /= time_avg[i]
                dat_avge[i][j,:] /= time_avge[i]

# average in frequency
for i in range(len(data)):
        dat_avg[i] = np.nanmean(dat_avg[i],axis=1)
        dat_avge[i] = np.nanmean(dat_avge[i],axis=1)        
        
# plot
j = 0
x = 128.*8.192e-6*384.*bst*np.arange(len(dat_avg[i]))
x -= np.mean(x)
x *= 15./60.
for i in np.arange(len(data)):
        if bname[i][0]==8:
                pylab.subplot(3,3,j+1)
                p1 = np.angle(dat_avg[i])
                #p2 = np.angle(dat_avge[i])
                plt.ylim(-90.,90.)
                plt.plot(x,(p1)*180./np.pi,'-',label=str(bname[i]))
                plt.xlabel('HA (arcmin)')
                if j==3:
                        plt.ylabel('Phase error (deg)')
                plt.legend()
                j += 1
                
plt.show()
plt.close()


