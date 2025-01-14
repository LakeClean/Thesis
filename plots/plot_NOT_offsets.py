import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

master_path = '/usr/users/au662080'

df = pd.read_csv(f'{master_path}/Speciale/data/NOT_order_file_log.txt')

directorys = df['directory'].to_numpy()
dates = df['date'].to_numpy()

folders = glob.glob(f'{master_path}/Speciale/data/target_analysis/*')

def mysort(x):
    y = x.split('_')
    return int(y[-3])
    


for folder in folders:
    print(folder)
    for date in dates:
        '''
        files = glob.glob(folder+f'/{date}/data/order*raw_spectrum.txt')
        files.sort(key=mysort)
        bin_wl =[]
        for file in files:
            df = pd.read_csv(file).to_numpy()
            bin_wl.append((df[-1,0]-df[0,0])/2)
        '''

        file = glob.glob(folder+f'/{date}/data/bf_fit_params.txt')
        
        if len(file)==0:
            continue

        df = pd.read_csv(file[0])
        rv = df['epoch_vrad1']
        #bin_wl = df['bin_wls']
        mean = np.median(rv[25:54])
        
        plt.vlines(25,-10,10,alpha=0.3,ls='--',color='black')
        plt.vlines(54,-10,10,alpha=0.3,ls='--',color='black')
        #plt.scatter(range(len(rv)),rv-mean)
        plt.scatter(range(len(rv)),rv-mean)




plt.xlabel('Wavelength of order bin [Å]',size=15)
plt.ylabel('Order',size=15)
plt.title('NOT: FIES (HIRES)',size=15)
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.show()
