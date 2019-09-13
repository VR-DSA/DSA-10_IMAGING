import numpy as np, matplotlib.pyplot as plt

bfac = 25
nl2 = 1000
bloc = 118
bnsamp = 4

t,XX = np.loadtxt('../tmp/ic_coh_XX.dat').transpose()
t,YY = np.loadtxt('../tmp/ic_coh_YY.dat').transpose()
XX = XX.reshape((256,1250))
YY = YY.reshape((256,1250))

#stds
st_XX = np.std(XX[0:100,:],axis=0)
st_YY = np.std(YY[0:100,:],axis=0)

# norm XX and YY
for i in range(256):
    XX[i,:] /= st_XX
    YY[i,:] /= st_YY

# find burst specs
bXX = np.sum(XX[bloc:bloc+bnsamp,:],axis=0)
bYY = np.sum(YY[bloc:bloc+bnsamp,:],axis=0)

# find Q and bin
bQ = bXX-bYY
bQ = np.mean(np.reshape(bQ,(1250/bfac,bfac)),axis=1)
bXX = np.mean(np.reshape(bXX,(1250/bfac,bfac)),axis=1)
bYY = np.mean(np.reshape(bYY,(1250/bfac,bfac)),axis=1)
#bQ = bQ/(bXX+bYY)

# grid on l2 axis
f = (1530.-np.arange(2048)*250./2048.)[350:1600]
f = np.mean(np.reshape(f,(1250/bfac,bfac)),axis=1)
fl2 = (2.998e8/(f*1e6))**2.
l2 = (np.linspace(2.998e8/(f[0]*1e6),2.998e8/(f[-1]*1e6),nl2))**2.
inQ = np.interp(l2,fl2,bQ)
inQ -= np.mean(inQ)

# plot
plt.plot(l2,inQ)
plt.show()

# fft
pspec = np.abs(np.fft.fft(inQ))[0:nl2/2]
plt.plot(pspec)
plt.show()

# print rm
rm = 2.5*np.pi/(l2.max()-l2.min())
print f[0],f[-1]
print 'RM =',rm



