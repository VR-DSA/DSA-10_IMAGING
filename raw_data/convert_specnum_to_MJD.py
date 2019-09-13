import numpy as np, sys

MJD = float(sys.argv[1])
specnum = float(sys.argv[2])

day = np.floor(MJD)
sec = (MJD-np.floor(MJD))*86400.
isec = int(np.round(sec))

tim = day + (1.*isec+524288.0*8.192e-6+specnum*8.192e-6+0.5)/86400.

print '%.10f'%(tim)


