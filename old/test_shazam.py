import shazam
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
#import template
template_dir = '/home/lakeclean/Documents/speciale/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
tfl_RG = template_data['arcturus']
tfl_MS = template_data['solarflux']
twl = template_data['wavelength']

lines = open('/home/lakeclean/Documents/speciale/file_log.txt').read().split('\n')
files = []
IDs = []
for line in lines[:-1]:
    line = line.split(',')
    file = line[2].strip()
    SEQID = line[1].strip()
    ID = line[0].strip()
    if SEQID == 'science':
        files.append(file)
        IDs.append(ID)

#filename = '/home/lakeclean/Documents/speciale/initial_data/2024-04-02/FIHd020137_step011_merge.fits'
#filename = '/home/lakeclean/Documents/speciale/initial_data/2024-04-13/FIHd130132_step011_merge.fits'
#filename = files[3]
for file,ID in zip(files,IDs):
    #analyse_spectrum(file,bin_size=200,use_SVD=True,show_bin_plots=False)
    if ID == 'KIC-9025370':
        filename = file
#KIC-9025370
#EPIC-246696804
data = pyfits.getdata(filename)
header = pyfits.getheader(filename)
lam = header['CDELT1']*np.arange(header['NAXIS1'])+header['CRVAL1']

epoch_name = header['TCSTGT'].strip(' ') #name of target
epoch_date = header['DATE'].strip(' ')   #date of fits creation

start_wl = 3800
end_wl = 10000
bin_size=100
n_bins = int((start_wl - end_wl)/bin_size)
if start_wl<lam[0]:
        start_wl = lam[0]
if end_wl > lam[-1]:
        end_wl = lam[-1]
n_bins = int(abs(start_wl - end_wl)/bin_size)
fig, ax = plt.subplots()

for i in range(n_bins):
    begin = start_wl + bin_size*i
    end = begin + bin_size
    index = np.where((lam>begin) & ( lam<end))[0]

    #raw
    wl = lam[index]
    fl = data[index]
    #norm
    wl, nfl           = shazam.normalize(wl,fl)
    wl, nfl           = shazam.crm(wl,nfl,q=[99.99])
    #flip and resamp
    rwl, rf_fl, rf_tl = shazam.resample(wl,nfl,twl,tfl_MS)
    #bf
    rvs, bf           = shazam.getBF(rf_fl,rf_tl,rvr=201,dv=1)
    #rvs, ccf = shazam.getCCF(rf_fl,rf_tl)
    #plot
    ax.plot(rvs,bf+i*1)

ax.set_xlabel('km/s')
ax.set_ylabel('broadening function')
ax.set_ylim(0,n_bins)
ax2 = ax.twinx()
ax2.set_ylabel('wavelength of bin')
ax2.set_ylim(start_wl+bin_size, end_wl-bin_size)
plt.show()



#for file,ID in zip(files,IDs):
#    analyse_spectrum(file,bin_size=500,use_SVD=True,show_bin_plots=True)








'''
#normalize
nwl, nfl = shazam.normalize(wl,fl)
fig, ax = plt.subplots()
ax.plot(nwl,nfl)


nwl, nfl = shazam.crm(nwl, nfl)

ax.plot(nwl,nfl,alpha=0.2)

plt.show()

r_wl, rf_fl, rf_tl = shazam.resample(nwl,nfl,twl,tfl_MS)

fig, ax = plt.subplots()
ax.plot(r_wl,rf_fl)
ax.plot(r_wl,rf_tl)
plt.show()

rvs,ccf = shazam.getCCF(rf_fl,rf_tl,rvr=401)
fig, ax = plt.subplots()
ax.plot(rvs,ccf)
plt.show()



rvs,bf = shazam.getBF(rf_fl,rf_tl,rvr=401)
fig, ax = plt.subplots()
ax.plot(rvs,bf)
plt.show()

rot = shazam.smoothBF(rvs,bf,sigma=5)

fig, ax = plt.subplots()
ax.plot(rot,bf)
plt.show()


#fit, _, _ = shazam.rotbf_fit(rvs,rot,30,res=90000,smooth=5,print_report=False)

#rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
'''
