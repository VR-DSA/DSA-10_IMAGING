from os import listdir
from os.path import isfile, join
import sys
from clip import fitsinfo
import numpy as np
import pyfits as pf

#
def is_inbetween(st0,st1,st):
	v0=np.array([np.cos(st0),np.sin(st0),0.0])
	v1=np.array([np.cos(st1),np.sin(st1),0.0])
	v=np.array([np.cos(st),np.sin(st),0.0])
	a=np.cross(v0,v)
	b=np.cross(v1,v)
	if a[2]>0 and b[2]<0:
		return True
	else:
		return False
#
def yield_all_inbetween(dirname,st,prefix,write2file=None):
	op=[]
	files = listdir(dirname)
	for f in files:
		if (isfile(join(dirname, f)) and f[-4:]=='fits' and prefix in f):
			(mjd0,mjd1,st0,st1)=fitsinfo('%s/%s'%(dirname,f),quiet=True)
			if is_inbetween(st0,st1,st):
				op.append([f,st0*180./np.pi/15.0,st1*180./np.pi/15.0])
	if write2file!=None:
		f=open("%s.txt"%write2file,'w')
		for fl in op:
			f.write("%s/%s\n"%(dirname,fl[0]))
		f.close()
	return op
#
#

def yield_cal_srcs(fname,catname="src_cat_bright.txt"):
	tp1,tp2,st0,st1=fitsinfo(fname,quiet=True)
	f=open(catname,'r')
	lines=f.readlines()
	f.close()
	names=[]
	ras=[]
	src_names=[]
	for line in lines:
		if line[0]!='#':
			w=line.strip('\n').split(',')
			names.append(w[0])
			ras.append(( float(w[1])+float(w[2])/60.0+float(w[3])/3600.0) *np.pi/12.0)
	for i in range(len(names)):
		if is_inbetween(st0,st1,ras[i]):
			if (ras[i]-st0)>1.5/60.0*15.*np.pi/180.0 and (st1-ras[i])>1.5/60.0*15.*np.pi/180.0:
				src_names.append(names[i])
	return list(set(src_names))	
 
def read_src_model(src_name,catname="src_cat_bright.txt"):
	src_model=[]
	f=open(catname,'r')
	lines=f.readlines()
	for line in lines:
	        w=line.strip('\n').split(',')
	        if w[0]==src_name:
	                ra=float(w[1])+float(w[2])/60.0+float(w[3])/3600.0
	                sgn = np.sign(float(w[4]))
	                dec=np.absolute(float(w[4]))+float(w[5])/60.0+float(w[6])/3600.0
	                dec*=sgn
	                src_model.append([ra*np.pi/12.0,dec*np.pi/180.0, float(w[7]), float(w[8]), float(w[9]),float(w[10])])
	if len(src_model)>0:
		return src_model
	else:
		print "Source "+src_name+" not found. This is fatal!"
		return None
#
