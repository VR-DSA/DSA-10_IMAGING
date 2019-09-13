import numpy as np, matplotlib.pyplot as plt, pyfits as pf, pylab
import sys
from const import *

vis_fname = sys.argv[1]

d = np.asarray(pf.open(vis_fname)[1].data).astype('float')
d = d[0:-1]
nt = len(d)/200/2048
d = d.reshape((nt,2048,200))
h = pf.open(vis_fname)[1].header
antennas = h['ANTENNAS'].split('-')

specs = np.swapaxes(np.mean(d[:,:,180:],axis=0),0,1)


for i in range(10):

    pylab.subplot(2,5,i+1)
    plt.plot(FREQ,10.*np.log10(specs[i*2]),'r')
    plt.plot(FREQ,10.*np.log10(specs[i*2+1]),'b')
    plt.title(antennas[i])

plt.show()

