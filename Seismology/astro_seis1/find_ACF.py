import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ps import powerspectrum
#from bottleneck import nanmedian, nanmean, nanmax, nanmin
#from scipy.interpolate import InterpolatedUnivariateSpline as INT
#pofrom scipy import optimize as OP
from matplotlib.colors import LogNorm
from matplotlib import cm
import glob
import seismology_functions as sf
from matplotlib.widgets import Button, Slider
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from peak_bagging_tool import simple_peak_bagging, mode
from scipy.ndimage import gaussian_filter
import lmfit
from scipy.integrate import quad
from ophobningslov import *
from scipy.optimize import minimize
from multiprocessing import Pool
import emcee
import corner




master_path = '/usr/users/au662080'

'''
This is simply done by computing the PDS with the use of the powerspectrum
class in from the ps module:
'''

#### Powerspectra: ####
#NOTICE: These appear to already have been filtered.


#importing log file
log_file_path = f'{master_path}/Speciale/data/Seismology/analysis/'
log_file_path += 'log_file.txt'
log_df = pd.read_csv(log_file_path)

IDs = log_df['ID'].to_numpy()
data_types = log_df['data_type'].to_numpy()
numax1_guesss = log_df['numax1_guess'].to_numpy()
numax2_guesss = log_df['numax2_guess'].to_numpy()
data_paths = log_df['data_path'].to_numpy()





#############################################################################
def save_data(ID,title,data,header):
    '''
    Give data as list of lists
    '''
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    path_to_save += title + '.txt'
    

    out_dict = {}
    for i, head in enumerate(header):
        out_dict[head] = data[i]

    #print(out_dict)
    out_df = pd.DataFrame(out_dict)
    #print(out_df)
    out_df.to_csv(path_to_save,index = False)


##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,
                  filtering = True):
    '''
    Function for finding numax and deltanu of given target.
    Also does a lot of plotting.
    Give data as power spectrum. and allow only for target with single visible
    seismic signal.
    '''
    
    
    ID_idx = np.where(ID == IDs)[0]
    if len(ID_idx) != 1:
        print('ID was given wrong')
        print('The ID should be among the following:')
        print(IDs)
        return 0

    data_type = data_types[ID_idx][0]
    numax1_guess = numax1_guesss[ID_idx][0]
    numax2_guess = numax2_guesss[ID_idx][0]
    data_path = data_paths[ID_idx][0]
    print('Analysing ',ID, '|  numax1 guess:', numax1_guess)
    if numax1_guess == numax2_guess: numax_guess = numax1_guess


    if data_type == 'dat': #data is timeseries format
        
        data_df = pd.read_csv(data_path,skiprows=13,delimiter=r"\s+").to_numpy()
        rel_time, flux, e_flux = data[:,0], data[:,1], data[:,2]
        idx = (0<e_flux)
        rel_time, flux, e_flux = rel_time[idx], flux[idx], e_flux[idx]
        PDS = powerspectrum(rel_time,flux,flux_err=e_flux,weighted=True)
        f, p0 = PDS.powerspectrum(scale='powerdensity')
        
        if plotting:
            fig,ax = plt.subplots()
            ax.plot(rel_time,flux)
            ax.set_title(ID)
            ax.set_xlabel('Truncated barycentric JD')
            ax.set_ylabel('Relative flux')
            plt.show()

    if data_type == 'pow': #data is power spectrum format
        data_df = pd.read_csv(data_path,skiprows=15,delimiter=r"\s+").to_numpy()
        f = data_df[:,0] #muHz
        p0 = data_df[:,1] #ppm**2 / muHz
        if saving_data: save_data(ID, 'power_spec', [f,p0],
                                  ['Frequency', 'power'])


    #### filtering: ####
    df = f[10]-f[9]
    win = int(1/df)
    if win%2==0: win+=1                              
    if filtering:
        #p_bg = sf.logmed_filter(f,p0,filter_width=0.2)
        #p = p0/p_bg
        p = p0
        p_filt = sf.filtering(p,win)
        if saving_data: save_data(ID, 'filt_power_spec', [f,p_filt],
                                  ['Frequency', 'filt_power'])


    #### Plotting filtered and unfiltered psd: 
    if plotting:
        fig, ax = plt.subplots()
        ax.plot(f,p0, label='raw')
        ax.plot(f,p_filt, label='Epan. filtered')
        
        ax.set_xlabel(r'frequency [$\mu$Hz]')
        ax.set_ylabel(r'power [ppm^2 / $\mu$Hz]')
        ax.set_title(f'Power spec of {ID}')
        ax.legend()
        plt.show()

    
    #### Finding ACF: ####
    env_scale=1
    sigma_env = env_scale * numax_guess/(4*np.sqrt(2*np.log(2)))

    HWHM = numax_guess/4
    idx1 = (f<numax_guess + 1.5*HWHM) & (f> numax_guess -1.5*HWHM)
    idx2 = (f<numax_guess + 3*HWHM) & (f> numax_guess -3*HWHM)

    weight = 1 / (sigma_env * np.sqrt(2*np.pi)) * np.exp(-(f - numax_guess)**2 / (2*sigma_env**2) )
    pds_w = p*weight

    acf1 = sf.autocorr_fft(p_filt[idx1])
    acf2 = sf.autocorr(p_filt[idx1])

    acf1_w = sf.autocorr_fft(pds_w[idx1])
    acf2_w = sf.autocorr(pds_w[idx1])

    df = f[10] - f[9]
    lagvec1 = np.arange(len(acf1))*df
    lagvec2 = np.arange(len(acf2))*df

    if saving_data: save_data(ID, 'ACF', [acf2,lagvec2],
                                  ['ACF', 'lagvec2'])
    if saving_data: save_data(ID, 'weighted_ACF', [acf2_w,lagvec2],
                                  ['ACF_w', 'lagvec2'])

    #Finding maximums in ACF:
    dnu_guess = sf.dnu_guess(numax_guess)
    dnu_peak1 = sf.find_maximum(dnu_guess,lagvec1,acf1/acf1[1],
                                dnu_guess*0.9,dnu_guess*1.1)[0]
    
    dnu_peak2 = sf.find_maximum(dnu_guess,lagvec2,acf2/acf2[1],
                                dnu_guess*0.9,dnu_guess*1.1)[0]
    print(f'dnu from unweighted PSD: {dnu_peak1}')
    print(f'dnu from weighted PSD: {dnu_peak2}')
    if saving_data: save_data(ID, 'dnu', [[dnu_peak1],[dnu_peak2]],
                                  ['dnu_peak1', 'dnu_peak2'])
    
                              


    smoothed = gaussian_filter(p[idx1], sigma=6)
    #plotting
    if plotting:
        fig,ax = plt.subplots(2,2)
        ax[0,0].set_title(ID)

        
        ax[0,0].plot(f[idx2], p[idx2])
        ax[0,0].plot(f[idx1], p[idx1])
        ax[0,0].plot(f[idx1],smoothed)
        ax[0,0].set_xlabel(r'frequency [$\mu$Hz]')
        ax[0,0].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


        ax[0,1].plot(f[idx2], pds_w[idx2])
        ax[0,1].plot(f[idx1], pds_w[idx1])
        ax[0,1].plot(f[idx2],np.max(pds_w[idx1])*weight[idx2]/np.max(weight),
                     color='k',ls='--',label='weight')
        ax[0,1].set_xlabel(r'frequency [$\mu$Hz]')
        ax[0,1].set_ylabel(r'power [$ppm^2 / \mu$Hz]')
        ax[0,1].legend()


        ax[1,0].plot(lagvec2, acf2/acf2[1])
        ax[1,0].plot(lagvec1, acf1/acf1[1])
        ax[1,0].set_xlabel(r'frequency lag [$\mu$Hz]')
        ax[1,0].set_ylabel(f'ACF')
        for i in range(7):
            i+=1
            ax[1,0].vlines(x=dnu_peak1*i,ymin=-1,ymax=1,ls='--',color='k')
            ax[1,0].vlines(x=dnu_peak2*i,ymin=-1,ymax=1,ls='--',color='k')


        ax[1,1].plot(lagvec2, acf2_w/acf2_w[1])
        ax[1,1].plot(lagvec1, acf1_w/acf1_w[1])
        ax[1,1].set_xlabel(r'frequency lag [$\mu$Hz]')
        ax[1,1].set_ylabel(f'ACF (weighted)')
        for i in range(7):
            i+=1
            ax[1,1].vlines(x=dnu_peak2*i,ymin=-1,ymax=1,ls='--',color='k')
            ax[1,1].vlines(x=dnu_peak2*i,ymin=-1,ymax=1,ls='--',color='k')

        fig.tight_layout()
        plt.show()




    




if True:
    analyse_power('KIC10454113',saving_data = True, plotting = True,
                  filtering = True) 

if False:  
    analyse_power('KIC9025370',saving_data = True, plotting = True,
                  filtering = True)

if False: 
    analyse_power('KIC12317678',saving_data = True, plotting = True,
                  filtering = True)

if False: 
    analyse_power('KIC4914923',saving_data = True, plotting = True,
                  filtering = True)

'''
if True:
    analyse_power('EPIC236224056',saving_data = True, plotting = False,
                  filtering = True)
if True:
    analyse_power('EPIC246696804',saving_data = True, plotting = False,
                  filtering = True)
if True:
    analyse_power('EPIC249570007',saving_data = True, plotting = False,
                  filtering = True)
if True:
    analyse_power('EPIC230748783',saving_data = True, plotting = False,
                  filtering = True)
if True:
    analyse_power('EPIC212617037',saving_data = True, plotting = False,
                  filtering = True)
'''











