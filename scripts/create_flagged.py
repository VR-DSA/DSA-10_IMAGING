import numpy as np, sys
sys.path.append('/mnt/nfs/code/dsaX_future/DSA-10/utils')
import plotNewVis as pv

fls = [sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]]
print 'Merging files'
pv.mergeVis(fls=fls,filename='data/merged.fits')
print 'Cleaning file...'
pv.cleanVis(fl='data/merged.fits',tbin=1000,thresh=4.0,plot=False,apply=True,filename='data/flagged.fits')

