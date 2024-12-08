import astropy.io.fits as pyfits
import pandas as pd
import matplotlib.pyplot as plt
from nsstools import NssSource
import make_table_of_target_info as mt
import numpy as np
import glob

#Plotting the templates to compare them

#Old template from idk where:
'''
template_dir = '/home/lakeclean/Documents/speciale/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
tfl_RG = template_data['arcturus']
tfl_MS = template_data['solarflux']
twl = template_data['wavelength']

path = '/home/lakeclean/Downloads/spvis.dat.gz'

df = pd.read_csv(path,delim_whitespace=True)

print(df.keys())

df = df.to_numpy()

plt.plot(10**(8)/df[:,0],df[:,1],color='r',label='(Reiners, 2016)')
plt.plot(twl,tfl_MS,color='b')
plt.show()
'''



#Comparing the values found by me and those from Gaia:

'''
nss = pd.read_csv("/home/lakeclean/Downloads/Jonatan.csv")
for i in range(4):
    source = NssSource(nss, indice=i)
    print(source.covmat())
    print(source.campbell())

tab = mt.get_table()
for i in tab['G_i'].data:
    print(np.degrees(i))
'''


#Showing that TNG red and blue is weird:

blue_path = '/home/lakeclean/Documents/speciale/initial_data/TNG/2016-07-19/KIC104.blue.norm.fits'
red_path = '/home/lakeclean/Documents/speciale/initial_data/TNG/2016-07-19/KIC104.red.norm.fits'
regular_path = '/home/lakeclean/Documents/speciale/target_analysis/KIC10454113/2016-07-19T22:19:33.321/data/order_*_normalized.txt'



red_data = pyfits.getdata(red_path)
red_header = pyfits.getheader(red_path)
red_lam = red_header['CDELT1']*np.arange(red_header['NAXIS1'])+red_header['CRVAL1']


blue_data = pyfits.getdata(blue_path)
blue_header = pyfits.getheader(blue_path)
blue_lam = blue_header['CDELT1']*np.arange(blue_header['NAXIS1'])+blue_header['CRVAL1']

plt.plot(red_lam+blue_lam[-1]-red_header['CRVAL1'],red_data,color='red')
plt.plot(blue_lam,blue_data,color='blue')


regular_files = glob.glob(regular_path)
for file in regular_files:
    df = pd.read_csv(file)
    regular_lam = df['wavelength [Å]']
    regular_data = df['flux (norm)']
    plt.plot(regular_lam,regular_data,color='orange')
    
plt.show()



















