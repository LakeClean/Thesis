import astropy.io.fits as pyfits
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

TNG_path = '/home/lakeclean/Documents/speciale/TNG_merged_file_log.txt'

TNG_df = pd.read_csv(TNG_path)
index = [0,1,2,6,7]
dates = TNG_df['date'].to_numpy()[index]
v_barys = TNG_df['v_bary'].to_numpy()[index]
print(TNG_df['ID'].to_numpy()[index])

print(v_barys)

'''
NOT_path = '/home/lakeclean/Documents/speciale/NOT_order_file_log.txt'

NOT_df = pd.read_csv(NOT_path)
'''

def wavelength_corr(wl,vbary=0):
    c = 299792 #speed of light km/s
    return (1+vbary/c)*wl

print(wavelength_corr(6080,0.5))

path = '/home/lakeclean/Documents/speciale/target_analysis/KIC10454113/'

TNG_files = []

for date in dates:

        TNG_files.append(path+f'{date.strip(' ')}/data/order_27_normalized.txt')
    

for file,v_bary in zip(TNG_files,v_barys):
    df = pd.read_csv(file)
    df = df.to_numpy()
    plt.plot(wavelength_corr(df[:,0],v_bary/1000),df[:,1])
    
plt.show()







    
    
