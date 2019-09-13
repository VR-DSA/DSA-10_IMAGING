import numpy as np
#C=299792458.0           # Speed of light in m/s
C=299706095.112 # in air
PI2=np.pi*2.0
PI=np.pi
HinSD=23.9344699		# Length of a sidereal day in hours
SinSD=23.9344699*3600.0		# in seconds
SinD=24.0*3600.0		# Seconds in a day
TSAMP=8.192e-6*128.*384.	# Sampling interval in s (0.2sec=factor of3 in prod.)
FREQ=1e6*(1530.0-np.arange(0.0,2048.0,1.0)*250.0/2048.0)
NF=150
F_START=350
F_END=1600
FREQ2=FREQ[F_START:F_END]
NF2=len(FREQ2)
FREQ1=1487.27539*1e6-(np.arange(625)+0.5)*2.*2.5e8/2048.
NF1=len(FREQ1)
LON=-118.283400
LAT=37.233386
NCORES=30
#POINTING=0.38422453855292943		# Declination of the pointing in radian
POINTING=1.2863076587		# Declination of the pointing in radian
#REF_EPOCH=57897.8841033
REF_EPOCH=58626.0
#REF_EPOCH=56900.0 # testing only
TIME_OFFSET=4.294967296			# Time error in cal data in seconds
MOFF=0./24. # error in MJD in days
