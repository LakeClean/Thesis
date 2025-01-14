import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/usr/users/au662080/Speciale/data/target_analysis/KIC10454113'
folders = glob.glob(f'{path}/2*')

folders = [x for x in folders if len(x) <len(path)+len('2017-06-28')+2]


for folder in folders:
    
    file = glob.glob(folder+'/data/bf_fit_params.txt')
    
    df = pd.read_csv(file[0])
    rv = df['epoch_vrad1']
    bin_wl = df['bin_wls']
    mean = np.mean(rv[11:14])
    
    plt.vlines(3900,-3,3,alpha=0.3,ls='--',color='black')
    plt.vlines(4845,-3,3,alpha=0.3,ls='--',color='black')
    plt.scatter(bin_wl,rv-mean)




plt.xlabel('Wavelength of order bin [Å]',size=15)
plt.ylabel('Radial Velocity - offset [km/s]',size=15)
plt.title('Keck: KIC10454113',size=15)
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.show()
