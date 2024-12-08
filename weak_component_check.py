import shazam
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import glob
#import pickle #For saving figures as matplotlib figures
from astropy.modeling import models, fitting
from time import time
from scipy.signal import find_peaks
from astropy.time import Time

'''
Idea: First shift one spectrum by the appropiate wavelength of the v_bary.
        Then shift the next spectrum and interpolate it to the points
        of the first spectra.
        repeat this with all spectra.
        The first spectra maybe also needs to be interpolated first.
        Lastly the interpolated spectra can be summed.
'''

#import template
template_dir = '/home/lakeclean/Documents/speciale/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
tfl_RG = template_data['arcturus']
tfl_MS = template_data['solarflux']
twl = template_data['wavelength']

#2024-05-26T22:58:27.122      #7
#2024-06-08T04:23:40.113      #8
#2024-09-01T21:57:52.339      #18
#2024-09-12T00:49:10.250      #19
dates = ['2024-05-26T22:58:27.122']
dates = ['2024-05-26T22:58:27.122','2024-09-01T21:57:52.339']
dates = ['2024-05-26T22:58:27.122','2024-09-12T00:49:10.250']
dates = ['2024-05-26T22:58:27.122','2024-06-08T04:23:40.113',
         '2024-09-01T21:57:52.339','2024-09-12T00:49:10.250']
for date in dates:
    print(Time(date).jd - 2457000)

path = '/home/lakeclean/Documents/speciale/initial_data'



path = f'/home/lakeclean/Documents/speciale/order_file_log.txt'
lines= open(path).read().split('\n')
all_IDs, all_vhelios,all_dates= [], [], []
all_vbary = {}
all_files = {}
for line in lines[:-1]:
    line = line.split(',')
    if line[1].strip() == 'science':
        if line[3].strip() in dates:
            all_IDs.append(line[0].strip())
            all_dates.append(line[3].strip())
            all_vhelios.append(float(line[5].strip()))
            all_files[line[3].strip()] = line[2].strip()
            all_vbary[line[3].strip()] = float(line[-1].strip())/1000 #correcting from m/s to km/s


datas = []
for date in dates:
    data, no_orders, bjd, vhelio, star, date, exp = shazam.FIES_caliber(all_files[f'{date}'])
    datas.append(data)


def wavelength_corr(wl,vbary=0):
    c = 299792 #speed of light km/s
    return (1+vbary/c)*wl

print(datas[0])
mean_of_spectra = np.zeros(2062)
start_order = 35
end_order = 60

for ID in all_IDs:
    if ID == 'KIC-4914923':
        data, no_orders, bjd, vhelio, star, date, exp = shazam.FIES_caliber(all_files[f'{date}'])
        for i in np.arange(start_order,end_order,1):
            #Pick out correct wl range
            wl, fl = shazam.getFIES(data,order=i)
            
            vbary = all_vbary[dates[0]]
            nwls = wavelength_corr(nwl,vbary)
            nfls = nfl
      
        

'''
normalize_bl = np.array([])
normalize_poly=1
normalize_gauss=True
normalize_lower=0.5
normalize_upper=1.5
                     
crm_iters = 1
crm_q = [99.0,99.9,99.99]
                     
resample_dv=1.0
resample_edge=0.0

getCCF_rvr=401
getCCF_ccf_mode='full'
                     
getBF_rvr=401
getBF_dv=1.0
                     
rotbf2_fit_fitsize=30
rotbf2_fit_res=60000
rotbf2_fit_smooth=2
rotbf2_fit_vsini1=10.0
rotbf2_fit_vsini2=5.0
rotbf2_fit_vrad1=-30.0
rotbf2_fit_vrad2=10.0
rotbf2_fit_ampl1=0.05
rotbf2_fit_ampl2=0.05
rotbf2_fit_print_report=False
rotbf2_fit_smoothing=True
                     
rotbf_fit_fitsize=30
rotbf_fit_res=60000
rotbf_fit_smooth=2.0

rotbf_fit_vsini=10.0
rotbf_fit_print_report=False
                     
use_SVD=True
SB_type=1
show_plots=True
save_plots=True
save_data = True
show_bin_plots=False
save_bin_info=False

ends = 100   

start_order = 35
end_order = 60
summed_bf = np.zeros(401)

                   
mean_of_spectra = np.zeros(2062)


for i in np.arange(start_order,end_order,1):

    #Pick out correct wl range
    wl, fl = shazam.getFIES(datas[0],order=i)
    

    #cutting of the ends:
    o_wl, fl = wl[ends:-ends], fl[ends:-ends]
    
    wl = np.linspace(o_wl[0],o_wl[-1],len(o_wl)*10)
    fl = np.interp(wl,o_wl,fl)
        
    #normalize
    nwl, nfl = shazam.normalize(wl,fl, normalize_bl,
                                    normalize_poly,normalize_gauss,
                                    normalize_lower,normalize_upper)


    nfl = nfl / np.median(nfl[np.where(np.percentile(nfl,95)<nfl)[0]])#The flux of the normalized flux that is above 95% percentile

    
    #Remove cosmic rays
    nwl, nfl = shazam.crm(nwl, nfl, crm_iters, crm_q)

    
    #Correct wavelength
    vbary = all_vbary[dates[0]]
    nwls = wavelength_corr(nwl,vbary)
    nfls = nfl
    
    for j in range(len(dates)-1):
        #Pick out correct wl range
        wl, fl = shazam.getFIES(datas[j+1],order=i)
        
        #cutting of the ends:
        wl, fl = wl[ends:-ends], fl[ends:-ends]

        
        #normalize
        nwl, nfl = shazam.normalize(wl,fl, normalize_bl,
                                    normalize_poly,normalize_gauss,
                                    normalize_lower,normalize_upper)
        nfl = nfl / np.median(nfl[np.where(np.percentile(nfl,99)<nfl)[0]])#The flux of the normalized flux that is above 95% percentile
        
        #Remove cosmic rays
        nwl, nfl = shazam.crm(nwl, nfl, crm_iters, crm_q)


        #Shift the spectrum
        vbary = all_vbary[dates[j+1]]
        nwl = wavelength_corr(nwl,vbary)
        
        nfls += np.interp(nwls,nwl,nfl)


    #normalize summed spectra:
    nwls, nfls = shazam.normalize(nwls,nfls, normalize_bl,
                                    normalize_poly,normalize_gauss,
                                    normalize_lower,normalize_upper)

    nfls = nfls / np.median(nfls[np.where(np.percentile(nfls,99)<nfls)[0]])#The flux of the normalized flux that is above 95% percentile

    
    

    #Resample and flip:
    r_wl, rf_fl, rf_tl = shazam.resample(nwls,nfls,twl,tfl_MS,
                                             resample_dv, resample_edge)


    #Get the bf
    rvs,bf = shazam.getBF(rf_fl,rf_tl,getBF_rvr, getBF_dv)

    #Fitting the bf
    fit, model, bfgs = shazam.rotbf_fit(rvs,bf,rotbf_fit_fitsize,
                                            rotbf_fit_res,rotbf_fit_smooth,
                                            rotbf_fit_vsini,rotbf_fit_print_report)

    summed_bf += bfgs


fig, ax  = plt.subplots()
ax.plot(rvs,summed_bf)
plt.show()
'''

'''
ends = 100   

start_order = 30
end_order = 60

all_summed = np.zeros(401)
for j in range(len(dates)):
    order_summed_bf = np.zeros(401)
    for i in np.arange(start_order,end_order,1):

    
        #Pick out correct wl range
        wl, fl = shazam.getFIES(datas[0],order=i)

        #cutting of the ends:
        o_wl, fl = wl[ends:-ends], fl[ends:-ends]

        wl = np.linspace(o_wl[0],o_wl[-1],len(o_wl)*2)
        fl = np.interp(wl,o_wl,fl)
        
        #normalize
        nwl, nfl = shazam.normalize(wl,fl, normalize_bl,
                                    normalize_poly,normalize_gauss,
                                    normalize_lower,normalize_upper)


        nfl = nfl / np.median(nfl[np.where(np.percentile(nfl,95)<nfl)[0]])#The flux of the normalized flux that is above 95% percentile


        #Remove cosmic rays
        nwl, nfl = shazam.crm(nwl, nfl, crm_iters, crm_q)

        #Shift spectrum
        vbary = all_vbary[dates[0]]
        nwls = wavelength_corr(nwl,vbary)
        nfls = nfl


        #Resample and flip:
        r_wl, rf_fl, rf_tl = shazam.resample(nwls,nfls,twl,tfl_MS,
                                             resample_dv, resample_edge)


        #Get the bf
        rvs,bf = shazam.getBF(rf_fl,rf_tl,getBF_rvr, getBF_dv)

        #Fitting the bf
        fit, model, bfgs = shazam.rotbf_fit(rvs,bf,rotbf_fit_fitsize,
                                            rotbf_fit_res,rotbf_fit_smooth,
                                            rotbf_fit_vsini,rotbf_fit_print_report)
        order_summed_bf += bfgs
    all_summed += order_summed_bf


fig, ax  = plt.subplots()
ax.plot(rvs,all_summed)
plt.show()
'''



