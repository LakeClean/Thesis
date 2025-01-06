import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 

df = pd.read_csv('/home/lakeclean/Documents/speciale/NOT_order_file_log.txt')

directorys = df['directory'].to_numpy()
dates = df['date'].to_numpy()

folders = glob.glob('/home/lakeclean/Documents/speciale/target_analysis/*')

for folder in folders:
    print(folder)
    for date in dates:

        
        file = glob.glob(folder+f'/{date}/data/bf_fit_params.txt')
        if len(file)==0:
            continue

        df = pd.read_csv(file[0])
        rv = df['epoch_vrad1']
        bin_wl = df['bin_wls']
        mean = np.mean(rv[11:14])
        
        #plt.vlines(3900,-3,3,alpha=0.3,ls='--',color='black')
        #plt.vlines(4845,-3,3,alpha=0.3,ls='--',color='black')
        plt.scatter(bin_wl,rv-mean)




plt.xlabel('Wavelength of order bin [Å]',size=15)
plt.ylabel('Radial Velocity - offset [km/s]',size=15)
#plt.title('Keck: KIC10454113',size=15)
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.show()
