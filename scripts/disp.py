import numpy as np, matplotlib.pyplot as plt, pyfits as pf, time

for i in range(24):

    a = pf.open('img_'+str(i)+'.fits')[0].data

    mad = 1.4826*np.median(np.abs(a-np.median(a)))
    print mad
    
    plt.imshow(a,vmin=-2.*mad,vmax=2.*mad)
    plt.title(str(i))
    plt.savefig('temp/img_'+str(i)+'.png',bbox_inches='tight')

    plt.close()



    

    
