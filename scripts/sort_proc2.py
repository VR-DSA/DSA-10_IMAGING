import numpy as np, sys, os

f1,f2,f3,f4,f5,specnum,snr,wid,dm,mjd,s1,s2 = np.loadtxt(sys.argv[1]).transpose()

wid = wid.astype('int')

n = len(f1)
for i in range(n):

    if s2[i] > 0.06/s1[i]:

        if s2[i]==50.:
            mvcmd = 'cp /var/www/html/CANDS/nan_'+str(s1[i])+'_'+str(snr[i])+'_'+str(dm[i])+'_'+str(wid[i])+'.png /var/www/html/CANDS2'
        else:
            mvcmd = 'cp /var/www/html/CANDS/'+str(s2[i])+'_'+str(s1[i])+'_'+str(snr[i])+'_'+str(dm[i])+'_'+str(wid[i])+'.png /var/www/html/CANDS2'
            
        print mvcmd
        os.system(mvcmd)


        
        
