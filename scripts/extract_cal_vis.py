import sys
#sys.path.append('/home/user/clip_21_July_2017')
from clip import extract_segment, predict_vis, delay_fit, remove_delay, block_avg, fitsinfo, noise_calc, LMsolve, flag_2d, plot_dynspec, extract_ac
from clip_bkeep import read_src_model, yield_all_inbetween, yield_cal_srcs
from const import *
import matplotlib.pyplot as plt, pylab
import pyfits as pf
from multiprocessing import Process as Proc1
import os


def output_vis(srcnam,A_vis,B_vis,mjds,bname):

        print "LENGTHS"
        print len(A_vis),len(bname)
        
        antennas="3-7-2-10-1-4-5-8-6-9"
        antennas_id=np.asarray([3,7,2,10,1,4,5,8,6,9])
        nt = A_vis[0].shape[0]
        nchan = A_vis[0].shape[1]
        print 'NT: ',nt,'NF: ',nchan
        
        # ordering of vis: [time, channel, baseline, pol, R/I]
        vis = np.zeros((nt,nchan,45,2,2)).astype('float')
        print 'PRE OUTPUT SHAPE: ',vis.shape
        print 'OUTPUT SHAPE: ',vis.ravel().shape
        bnames = np.zeros((45,2)).astype('int')
        ii=0
        for i in np.arange(0,9):
                for j in np.arange(i+1,10):
                        bnames[ii,0]=antennas_id[j]-1
                        bnames[ii,1]=antennas_id[i]-1
                        ii+=1

        print bname,bnames
        for i in np.arange(len(A_vis)):

                # find baseline id
                bid=-1
                print 'Finding matching baseline'
                for j in np.arange(45):
                        if bname[i][0]==bnames[j,0]:
                                if bname[i][1]==bnames[j,1]:
                                        print bname[i],bnames[j,:]
                                        bid=j

                # fill vis array
                if (bid!=-1):
                        print 'filling vis array'
                        for nti in np.arange(nt):
                                for nchani in np.arange(nchan):
                                        vis[nti,nchani,bid,0,0] = A_vis[i][nti,nchani].real
                                        vis[nti,nchani,bid,0,1] = -A_vis[i][nti,nchani].imag
                                        vis[nti,nchani,bid,1,0] = B_vis[i][nti,nchani].real
                                        vis[nti,nchani,bid,1,1] = -B_vis[i][nti,nchani].imag


        vis=vis.ravel()
        np.savez('/home/user/tmp/TST/output_'+srcnam+'.npz',vis=vis,ANTENNAS=antennas,NAXIS2=nt,TSAMP=0.4026532,MJD=mjds)
        
                
        
                

def extract_vis(src,pol,quiet,fl,usePredict=False,raoffset=0.0):

        # fill antennas and baseline lengths
        aname = pf.open(fl,ignore_missing_end=True)[1].header['ANTENNAS'].split('-')
        nchan = pf.open(fl,ignore_missing_end=True)[1].header['NCHAN']
        tp=np.loadtxt('antpos_ITRF.txt')
        fqs=pf.open(fl,ignore_missing_end=True)[1].header['FCH1']*1e6-(np.arange(nchan)+0.5)*2.*2.5e8/2048.

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
        data,st,mjd,Imid=extract_segment(fname=fl,stmid=stmid,seg_len=5./60.0/24.0*PI2, pol=pol, quiet=False)

        nsamps = data[0].shape[0]
        if usePredict:
                model=predict_vis(blen,mjd,fqs,src_model=src_model,pointing=POINTING,quiet=quiet)	# Predict the visibilities	
                return model,mjd,bname

        return data,mjd,bname


# Predefinitions
quiet=True
fl=sys.argv[1]
srcnam=sys.argv[2]
usePredict=False
raoffset=(27./60./24.)*2.*np.pi

A_vis,mjds,bname = extract_vis(srcnam,'A',quiet,fl,usePredict=usePredict,raoffset=raoffset)
B_vis,mjds,bname = extract_vis(srcnam,'B',quiet,fl,usePredict=usePredict,raoffset=raoffset)
                
output_vis(srcnam,A_vis,B_vis,mjds,bname)
