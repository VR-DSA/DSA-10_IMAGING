import sys
from clip import extract_segment, predict_vis, delay_fit, remove_delay, block_avg, fitsinfo, noise_calc, LMsolve, flag_2d, plot_dynspec, extract_ac
from clip_bkeep import read_src_model, yield_all_inbetween, yield_cal_srcs
from const import *
import matplotlib.pyplot as plt, pylab
import pyfits as pf
from multiprocessing import Process as Proc1
import os



# Predefinitions
freq_cadence=25 # average to 25 chans per sub-band for calibration
quiet=True
bst=100		# Block size for time avg and flagging
bsf=100		# Block size for freq flagging
src = sys.argv[3]#'J1200+7300'#'J1927+7358'
fl = sys.argv[1]
pol = sys.argv[2]
raoffset = 27.*(PI2/60./24.) # rad
seglen = 6. # min
refant = 6 # antenna 7

# 0 - make initial guess including delays.
#1 - average with calF with weight WT. use calF delays.
#2 - only derive phase correction to calF.
task = int(sys.argv[4])
if task>0:
        calF = sys.argv[5]
if task==1:
        WT = float(sys.argv[6])

if task>0:
        cal = np.load(calF)

# fill antennas and baseline lengths
aname = pf.open(fl,ignore_missing_end=True)[1].header['ANTENNAS'].split('-')
nchan = pf.open(fl,ignore_missing_end=True)[1].header['NCHAN']
tp=np.loadtxt('antpos_ITRF.txt')
fqs=pf.open(fl,ignore_missing_end=True)[1].header['FCH1']*1e6-(np.arange(nchan)+0.5)*2.*2.5e8/2048.
print nchan

blen=[]
bname=[]
for i in np.arange(9)+1:
        for j in np.arange(i):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                bname.append([a1,a2])
                blen.append(tp[a1,1:]-tp[a2,1:])

print "Antennas: "+str(aname)
print "Baseline: "+str(bname)
print "Source "+str(src)


src_model=read_src_model(src)
stmid=src_model[0][0]+raoffset
data=[]

# READ DATA
data,st,mjd,Imid=extract_segment(fname=fl,stmid=stmid,seg_len=seglen/60.0/24.0*PI2, pol=pol, quiet=False)

nsamps = data[0].shape[0]
model=predict_vis(blen,mjd,fqs,src_model=src_model,pointing=POINTING,quiet=quiet)	# Predict the visibilities	
dbym=[]	# data by model

f = block_avg(np.reshape(fqs,(1,len(fqs))),bst=1,bsf=freq_cadence).flatten()	

	
for i in range(len(data)):
	dbym.append((data[i]/model[i])/np.abs(data[i]/model[i]))


# measure delays
if task==0:
        
        # plot and fix delays
        taus,sigs = delay_fit(dbym,bname,Imid,fqs,basename="/home/user/tmp/fringe")
        ant_delays = np.zeros(10)

        for i in range(len(aname)):
                if i==refant:
                        ant_delays[i] = 0.0
                else:
                        try:
                                ant_delays[i] = taus[bname.index([i,refant])]
                        except:
                                ant_delays[i] = -taus[bname.index([refant,i])]


        print "antenna delays = " + str(ant_delays)
        b_delays = np.zeros((len(bname)))
        for i in range(len(bname)):
                b_delays[i] = ant_delays[bname[i][0]]-ant_delays[bname[i][1]]

# remove delays
if task>0:
        ant_delays = cal['ant_delays']
        b_delays = cal['b_delays']

remove_delay(dbym,b_delays,fqs)


# calibrate bandpass or phase

# average data to bsf
dat_avg = []
for i in range(len(data)):
        dat_avg.append(block_avg(dbym[i],bst=nsamps,bsf=freq_cadence))
        dat_avg[i] = dat_avg[i].flatten()
gains = []
for i in range(len(aname)):
        gains.append(np.zeros(dat_avg[0].shape,dtype=np.complex64))

if task<2:
        
        # calibrate
        gains[refant] += 1.+0.*1j
        for i in range(len(bname)):
                a1 = bname[i][0]
                a2 = bname[i][1]
                ply = np.conj(dat_avg[i])
                if a1==refant:
                        gains[a2] = ply
                if a2==refant:
                        gains[a1] = np.conj(ply)

        for i in np.arange(len(aname)):
                gains[i] /= np.abs(gains[i])

        if task==1:
                for i in np.arange(len(aname)):
                        gains[i] = WT*gains[i] + (1.-WT)*cal['gains'][i]

        for i in np.arange(len(aname)):
                gains[i] /= np.abs(gains[i])

                        
if task==2:

        # apply gains and derive phase offset
        gains[refant] += 1.+0.*1j
        for i in range(len(bname)):
                a1 = bname[i][0]
                a2 = bname[i][1]
                ply = np.conj(dat_avg[i])
                if a1==refant:
                        gains[a2] = ply
                if a2==refant:
                        gains[a1] = np.conj(ply)

        for i in np.arange(len(aname)):
                gains[i] /= np.abs(gains[i])
                mgain = np.mean(gains[i]/cal['gains'][i])
                gains[i] = cal['gains'][i]*mgain
                gains[i] /= np.abs(gains[i])
        

# save calibs
sav_fl = '/home/user/tmp/'+pol+'cal.npz'
np.savez(sav_fl,ant_delays=ant_delays,b_delays=b_delays,src_name=src,src_model=src_model,gains=gains,pol=pol,aname=aname,freq_cadence=freq_cadence)
