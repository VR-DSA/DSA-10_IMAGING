import sys
import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt, pylab

# arguments: input fits name

burst_nsamp=1
vis_fname=sys.argv[1]
oname=sys.argv[2]

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
t1a = np.zeros(nt)
t2a = np.zeros(nt)
t3a = np.zeros(nt)

print 'Have nt: ',nt
for i in np.arange(5):
        ts = np.mean(alld,axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        tts += ts/5.

        ts = np.mean(alld[:,0:550,:],axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        t1 += ts/5.

        ts = np.mean(alld[:,225:550,:],axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        t1a += ts/5.
        
        ts = np.mean(alld[:,550:1100,:],axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        t2 += ts/5.

        ts = np.mean(alld[:,550+225:1100,:],axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        t2a += ts/5.
        
        ts = np.mean(alld[:,1100:,:],axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        t3 += ts/5.

        ts = np.mean(alld[:,1300:,:],axis=1)
        ts = np.mean(ts[:,180+4*i:180+4*(i+1)],axis=1)-np.mean(ts[:,180+4*i:180+4*(i+1)])
        t3a += ts/5.
        
        
burst_loc = np.where(tts==np.max(tts))[0][0]

# decide whether to continue
S1_mn = (np.max(tts)-np.mean(tts))/np.std(tts)
std = 1.4826*np.median(np.abs(tts-np.median(tts)))
S1 = (np.max(tts)-np.median(tts))/std

# print STATs
print 'STATS',S1

# make plot of TS
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(tts,'k-')
plt.plot(np.asarray([burst_loc]),np.asarray([tts[burst_loc]]),'ro')
plt.title('MED: '+str(S1)+'; MEAN: '+str(S1_mn))
px = np.arange(2)*len(tts)
py = np.zeros(2)
plt.plot(px,py+np.mean(tts),'r--')
plt.plot(px,py+np.mean(tts)+np.std(tts),'r--')
plt.plot(px,py+np.median(tts),'g-')
plt.plot(px,py+np.median(tts)+std,'g-')

plt.subplot(212)
tts = tts/tts.max()
plt.plot(tts,'k-')
plt.plot(np.asarray([burst_loc]),np.asarray([tts[burst_loc]]),'ro')
t1 = t1/t1.max()-1.
t1a = t1a/t1a.max()-1.
plt.plot(t1,'b-')
#plt.plot(t1a,'b--')
t2 = t2/t2.max()-1.
t2a = t2a/t2a.max()-1.
plt.plot(t2,'g-')
#plt.plot(t2a,'g--')
t3 = t3/t3.max()-1.
t3a = t3a/t3a.max()-1.
plt.plot(t3,'r-')
#plt.plot(t3a,'r--')
plt.savefig(oname+'.png',bbox_inches='tight')
plt.close()

