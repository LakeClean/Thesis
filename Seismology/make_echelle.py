import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ps import powerspectrum
from bottleneck import nanmedian, nanmean, nanmax, nanmin
from scipy.interpolate import InterpolatedUnivariateSpline as INT
from scipy import optimize as OP
from matplotlib.colors import LogNorm
from matplotlib import cm
import glob
import seismology_functions as sf

'''
Data is given in either filtered Power spectra or as filtered timeseries.
We first deal with the timeseries:

This is simply done by computing the PDS with the use of the powerspectrum
class in from the ps module:
'''

#Folder for all data
folder_path = '/home/lakeclean/Documents/speciale/Seismology/data/'


#### Timeseries: ####
#importing data:
timeseries_files = glob.glob(folder_path + '*.dat')

print(timeseries_files)

numaxs = [76.48,50.41,25.35] #guess at numax
#guess_dnus = [1,2.2,0.4] #guesses at dnu
for time_file,numax in zip(timeseries_files[0:],numaxs[0:]):
    
    
    lines = open(time_file).read().split('\n')
    ID = lines[3]
    print('Analysing ',ID, '|  numax guess:', numax)

    #plotting the time series
    data = pd.read_csv(time_file,skiprows=13,delimiter=r"\s+").to_numpy()
    rel_time, flux, e_flux = data[:,0], data[:,1], data[:,2]
    idx = (0<e_flux)
    rel_time, flux, e_flux = rel_time[idx], flux[idx], e_flux[idx]

    if False:
        fig,ax = plt.subplots()
        ax.plot(rel_time,flux)
        ax.set_title(ID)
        ax.set_xlabel('Truncated barycentric JD')
        ax.set_ylabel('Relative flux')
        plt.show()

    
        

    #Finding PDS:
    PDS = powerspectrum(rel_time,flux,flux_err=e_flux,weighted=True)
    f_raw, p_raw = PDS.powerspectrum(scale='powerdensity')
    idx_limit = (f_raw<2*numax)
    f, p0 = f_raw[idx_limit], p_raw[idx_limit]

    #plotting raw:
    if True:
        fig,ax = plt.subplots()
        ax.plot(f_raw,p_raw)
        ax.set_title(ID)
        ax.set_xlabel('PSD')
        ax.set_ylabel('Frequency')
        plt.show()
    

    p_bg = sf.logmed_filter(f, p0, filter_width=0.2)
    p = p0/p_bg

    smo_width = 1
    df = f[10] - f[9]
    win = int(smo_width/df)
    if win%2==0: win+=1

    p_filt = sf.filtering(p,win)
    
    #### plotting full powerspectrum: ####
    if True:
        fig, ax = plt.subplots()
        ax.plot(f_raw,p_raw,label='raw psd')
        ax.plot(f,p,label='logmed filtered')
        ax.plot(f,p_filt,label='Epanechnikov filtered')
        ax.set_xlabel(r'frequency [$\mu$Hz]')
        ax.set_ylabel(r'power [ppm^2 / $\mu$Hz]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(ID)
        ax.legend()
        plt.show()

    #### Finding ACF: ####
    env_scale=1
    sigma_env = env_scale * numax/(4*np.sqrt(2*np.log(2)))
    HWHM = numax/4
    idx1 = (f<numax + 1.5*HWHM) & (f> numax -1.5*HWHM)
    idx2 = (f<numax + 3*HWHM) & (f> numax -3*HWHM)


    
    weight = 1 / (sigma_env * np.sqrt(2*np.pi)) * np.exp(-(f - numax)**2 / (2*sigma_env**2) )
    pds_w = p*weight

    acf1 = sf.autocorr_fft(p_filt[idx1])
    acf2 = sf.autocorr(p_filt[idx1])


    acf1_w = sf.autocorr_fft(pds_w[idx1])
    acf2_w = sf.autocorr(pds_w[idx1])

    lagvec1 = np.arange(len(acf1))*df
    lagvec2 = np.arange(len(acf2))*df

    #Finding maximums in ACF:
    dnu_guess = sf.dnu_guess(numax)
    dnu_peak1 = sf.find_maximum(dnu_guess,lagvec1,acf1/acf1[1],dnu_guess*0.9,dnu_guess*1.1)[0]
    dnu_peak2 = sf.find_maximum(dnu_guess,lagvec2,acf2/acf2[1],dnu_guess*0.9,dnu_guess*1.1)[0]
    

    #plotting
    fig,ax = plt.subplots(2,2)

    ax[0,0].plot(f[idx2], p_filt[idx2])
    ax[0,0].plot(f[idx1], p_filt[idx1])
    ax[0,0].set_xlabel(r'frequency [$\mu$Hz]')
    ax[0,0].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


    ax[0,1].plot(f[idx2], pds_w[idx2])
    ax[0,1].plot(f[idx1], pds_w[idx1])
    #ax[0,1].plot(f[idx2],np.max(pds_w[idx1])*weight[idx2]/np.max(weight),color='k',ls='--')
    ax[0,1].set_xlabel(r'frequency [$\mu$Hz]')
    ax[0,1].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


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
    ax[1,1].set_ylabel(f'ACF')

    fig.tight_layout()
    plt.show()

    # Calculating echelle diagram:
    FWHM = numax/2
    start = numax-FWHM
    stop = numax+FWHM

    A,B,EchelleVals=sf.Echellediagram(p,f,dnu_peak1,start,
                                      stop,no_repeat=1)
    VMIN = np.min(EchelleVals)
    VMAX = np.max(EchelleVals)

    #plot echelle

    fig, ax = plt.subplots()
    ax.imshow(EchelleVals,aspect='auto',
              extent=[0,dnu_peak1,start,np.max(B)+0.5*dnu_peak1],
              norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
              cmap=cm.gray_r,zorder=0)
    ax.set_xlabel(f' Frequency mod dnu={np.round(dnu_peak1,2)}muHz')
    ax.set_ylabel('Frequency muHz')
    ax.set_title(ID)



    plt.show()





#### Powerspectra: ####
#NOTICE: These appear to already have been filtered.

#importing
power_files = glob.glob(folder_path + '*.pow')
numaxs = [3000,2250,1200,1800] #guesses at numax
#guess_dnus = [132,100,100,100,100] #guesses at dnu
for power_file,numax in zip(power_files[0:],numaxs[0:]):
    
    lines = open(power_file).read().split('\n')
    ID = lines[2]
    print('Analysing ',ID, '|  numax guess:', numax)# '|  dnu guess:', dnu_guess)
    data = pd.read_csv(power_file,skiprows=13,delimiter=r"\s+").to_numpy()

    f = data[:,0] #muHz
    p0 = data[:,1] #ppm**2 / muHz

    #### plotting full powerspectrum: ####
    if True:
        fig, ax = plt.subplots()
        ax.plot(f,p0)
        ax.set_xlabel(r'frequency [$\mu$Hz]')
        ax.set_ylabel(r'power [ppm^2 / $\mu$Hz]')
        ax.set_title(f'Raw data: {ID}')
        plt.show()


    #### filtering: ####
    df = f[10]-f[9]
    win = int(1/df)
    if win%2==0: win+=1
    p_bg = sf.logmed_filter(f,p0,filter_width=0.2)
    p = p0/p_bg
    p_filt = sf.filtering(p,win)


    #plotting filtered psd: 
    if True:
        fig, ax = plt.subplots()
        ax.plot(f,p_filt)
        ax.set_xlabel(r'frequency [$\mu$Hz]')
        ax.set_ylabel(r'power [ppm^2 / $\mu$Hz]')
        ax.set_title(f'Filtered data: {ID}')
        plt.show()

    
    #### Finding ACF: ####
    env_scale=1
    sigma_env = env_scale * numax/(4*np.sqrt(2*np.log(2)))

    HWHM = numax/4
    idx1 = (f<numax + 1.5*HWHM) & (f> numax -1.5*HWHM)
    idx2 = (f<numax + 3*HWHM) & (f> numax -3*HWHM)

    weight = 1 / (sigma_env * np.sqrt(2*np.pi)) * np.exp(-(f - numax)**2 / (2*sigma_env**2) )
    pds_w = p*weight

    acf1 = sf.autocorr_fft(p_filt[idx1])
    acf2 = sf.autocorr(p_filt[idx1])


    acf1_w = sf.autocorr_fft(pds_w[idx1])
    acf2_w = sf.autocorr(pds_w[idx1])

    df = f[10] - f[9]
    lagvec1 = np.arange(len(acf1))*df
    lagvec2 = np.arange(len(acf2))*df

    #Finding maximums in ACF:
    dnu_guess = sf.dnu_guess(numax)
    dnu_peak1 = sf.find_maximum(dnu_guess,lagvec1,acf1/acf1[1],dnu_guess*0.9,dnu_guess*1.1)[0]
    dnu_peak2 = sf.find_maximum(dnu_guess,lagvec2,acf2/acf2[1],dnu_guess*0.9,dnu_guess*1.1)[0]
    

    #plotting
    if True:
        fig,ax = plt.subplots(2,2)
        ax[0,0].set_title(ID)

        ax[0,0].plot(f[idx2], p[idx2])
        ax[0,0].plot(f[idx1], p[idx1])
        ax[0,0].set_xlabel(r'frequency [$\mu$Hz]')
        ax[0,0].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


        ax[0,1].plot(f[idx2], pds_w[idx2])
        ax[0,1].plot(f[idx1], pds_w[idx1])
        #ax[0,1].plot(f[idx2],np.max(pds_w[idx1])*weight[idx2]/np.max(weight),color='k',ls='--')
        ax[0,1].set_xlabel(r'frequency [$\mu$Hz]')
        ax[0,1].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


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

        fig.tight_layout()
        plt.show()

    # Calculating echelle diagram:
    FWHM = numax/2
    start = numax-FWHM
    stop = numax+FWHM

    A,B,EchelleVals=sf.Echellediagram(p,f,dnu_peak1,start,
                                      stop,no_repeat=1)
    VMIN = np.min(EchelleVals)
    VMAX = np.max(EchelleVals)

    #plot echelle
    if True:
        fig, ax = plt.subplots()
        ax.imshow(EchelleVals,aspect='auto',
                  extent=[0,dnu_peak1,start,np.max(B)+0.5*dnu_peak1],
                  norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
                  cmap=cm.gray_r,zorder=0)
        ax.set_xlabel(f' Frequency mod dnu={np.round(dnu_peak1,2)}muHz')
        ax.set_ylabel('Frequency muHz')
        ax.set_title(ID)

        plt.show()


    #### Collapsed power: ####
    idx_limit = (f<2*numax)
    f, p_filt = f[idx_limit], p[idx_limit]
    Norders = 5
    ncollap = int(dnu_peak1/df)
    collapse = np.zeros(ncollap)
    start = np.argmin(np.abs(f-(numax-dnu_peak1*Norders/2)))
    for j in range(Norders):
        collapse += p_filt[start+ncollap*j:start+ncollap*(j+1)]
    collapsed_normed = collapse/Norders
    eps_axis = sf.crange2(1,2,
                         f[start:start+ncollap]/dnu_peak1 \
                         - int(f[start]/dnu_peak1) +1)
    idx_sort = np.argsort(eps_axis)
    eps_axis = (eps_axis[idx_sort]-1)*dnu_peak1
    collapsed_normed = collapsed_normed[idx_sort]
    if False:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(f,p_filt)
        ax[1].plot(eps_axis,collapsed_normed)
        ax[1].set_xlabel('Frequency')
        plt.show()


    #### ridge_centroids: ####
    '''
    f = data[:,0] #muHz
    p = data[:,1] #ppm**2 / muHz
    p_bg = sf.logmed_filter(f,p,filter_width=0.2)
    p = p/p_bg
    
    smo_width = 5
    df = f[10]-f[9]
    win = int(smo_width/df)
    if win%2==0:
        win+=1
    p_filt = sf.filtering(p,win)
    
    start_fre = numax*0.6
    stop_fre = numax*1.4
    dnus = np.linspace(dnu_peak1-2*dnu_peak1/100,dnu_peak1+2*dnu_peak1/100, 7)
    oaps_fre, av_oaps, oaps = sf.OAPS(f,p_filt,start_fre,
                                      stop_fre,dnus,K=4)
    fig,ax = plt.subplots(2,1)
    for k in range(oaps.shape[0]):
        ax[0].plot(oaps_fre,oaps[k,:])
        
    ax[1].plot( oaps_fre,av_oaps)
    
    av_oaps_filt = filtering(av_oaps,win)
    FWHM = numax/2
    start = numax-FWHM
    stop = numax+FWHM
    A,B,Echellediagram(p_filt,f,dnu_peak1,start,stop,no_repreat=1)
    fig,ax = plt.subplots()
    ax.imshow(EchelleVals,aspect='auto',
                  extent=[0,dnu_peak1,start,np.max(B)+0.5*dnu_peak1],
                  norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
                  cmap=cm.gray_r,zorder=0)

    for i in range(len(B)):
        try:
            idx_cut = (oaps_fre>B[i]-dnu_peak1/2) & (oaps_fre<B[i]+dnu_peak1/2)
            oap_fcut, oap_valcut = oaps_fre[idx_cut], av_oaps_filt[idx_cut]
            eps_vals = np.mod(oap_fcut,dnu_peak1)/dnu_peak1 + 1
            idx_cut_l0 = (eps_vals<1.5)
            f0 = oap_fcut[idx_cut_l0][np.argmax(oap_valcut[idx_cut_l0])]
            idx_cut_l1 = (eps_vals>1.5)
            f1 = oap_fcut[idx_cut_l1][np.argmax(oap_valcut[idx_cut_l1])]
            idx_cut_l2 = (eps_vals<1.5) & (oap_fcut<f0-0.8)
            f2 = oap_fcut[idx_cut_l2][np.argmax(oap_valcut[idx_cut_l2])]
            ax.scatter(np.mod(f1,dnu_peak1),f1,color='C1')
            ax.scatter(np.mod(f2,dnu_peak1),f2,color='C2')
            ax.scatter(np.mod(f0,dnu_peak1),f0,color='C0')
        except (IndexError, ValueError):
            continue

            
    plt.show()
    '''
    
    


'''
# Calculating echelle diagram:
df = pd.read_csv(datasets[0],skiprows=13,delimiter=r"\s+").to_numpy()

f = df[:,0] #muHz
p = df[:,1] #ppm**2 / muHz
FWHM = numax/2
start = numax-FWHM
stop = numax+FWHM

A,B,EchelleVals=sf.Echellediagram(p,f,guess_dnus[0],start,stop,no_repeat=1)
VMIN = np.min(EchelleVals)
VMAX = np.max(EchelleVals)

#plot spectrum
fig, ax = plt.subplots()

idx_plot = (f>start) & (f<stop)

ax.plot(f[idx_plot],p[idx_plot])

#plot echelle

fig, ax = plt.subplots()
ax.imshow(EchelleVals,aspect='auto',extent=[0,guess_dnus[0],start,
                                            np.max(B)+0.5*guess_dnus[0]],
          norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
          cmap=cm.gray_r,zorder=0)



plt.show()
'''
