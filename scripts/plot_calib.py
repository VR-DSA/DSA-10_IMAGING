import numpy as np, matplotlib.pyplot as plt
import pylab

f1 = np.load('Bcal_J1200_noDel.npz')
f2 = np.load('Bcal_J1459_noDel.npz')

g1 = f1['gains']
g2 = f2['gains']

plt.figure(figsize=(10.,7.))
for i in range(10):
    pylab.subplot(2,5,i+1)
    p1 = np.angle(g1[i]/g2[i])
    #p2 = np.angle(g2[i])
    plt.ylim(-180.,180.)
    plt.plot(p1*180./np.pi,'-',label='Ant '+str(i+1))
    #plt.plot(p2*180./np.pi,'-',label='Ant '+str(i+1)+' J1459')
    plt.legend()

plt.show()


                            
