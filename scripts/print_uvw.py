import sys
from clip import calc_uvw
import numpy as np
import matplotlib.pyplot as plt

# fill antennas and baseline lengths
tp=np.loadtxt('antpos_ITRF.txt')
aname = ['3','7','2','10','1','4','5','8','6','9']

blen=[]
for i in np.arange(9)+1:
        for j in np.arange(i):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                blen.append(tp[a1,1:]-tp[a2,1:])


u = np.zeros(len(blen))
v = np.zeros(len(blen))
w = np.zeros(len(blen))

for i in range(len(u)):
    u[i],v[i],w[i] = calc_uvw(blen[i],[58000.0])
    print i,u[i],v[i],w[i]

plt.scatter(u,v)
plt.show()

    
