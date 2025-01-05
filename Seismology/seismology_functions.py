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

#############
#   All or most functions are developed by Mikkel and are from the
#   Advanced stellar evolution course.
############

#########################################################################
# Code examples from Mikkel (timeseries):
#########################################################################

#### We generate a sine wave: #### 
def sine_wave(time,A=1, f=1, phi=np.pi/2):
    return A*np.sin(2*np.pi*f*time + phi)

#### simple filtering: #### 
def filtering(signal,win,filt='Epanechnikov'):
    '''
    signal: signal to be filtered
    win: window width in number of points:
    filt: the type of filter (window) to be used
    '''

    def Epanechnikov(win):
        u = np.array(range(win)) - np.array(range(win)).max()/2
        u /= np.array(range(win)).max()/2
        return 3/4 * (1 - u**2)

    def Tricube(win):
        u = np.array(range(win)) - np.array(range(win)).max()/2
        u /= np.array(range(win)).max()/2
        return 70/81 * (1 - abs(u)**3)**3
    
    def Uniform(win):
        u = np.array(range(win)) - np.array(range(win)).max()/2
        u /= np.array(range(win)).max()/2
        return 1/2
        
        
    filters = ['Epanechnikov', 'Tricube', 'Uniform']
    options = [Epanechnikov,Tricube,Uniform]
    for i in range(len(filters)):
        if filters[i]==filt:
            window = options[i]
            break
            
    w = window(win)



    sig = np.r_[signal[int(win)-1:0:-1], signal, signal[-1:-int(win):-1] ]
    y = np.convolve(w/w.sum(),sig,mode='valid')

    y = y[int(np.floor(win/2)):len(y)-int(np.floor(win/2))]

    return y

#### median filter ####
def median_filter(time, flux, window_size):
    """
    Apply a median filter to the flux data.

    Parameters:
    -time: np.array
    -flux_ np.array
    -window_size: int must be odd

    returns:
    numpy array: The filtered flux data
    """
    if window_size %2 ==0:
        raise ValueError('Window size must be odd')
    half_window = window_size //2
    flux_padded = np.pad(flux, (half_window, half_window), mode='edge')
    flux_filtered = np.zeros = np.zeros_like(flux)

    for i in range(len(flux)):
        flux_filtered[i] = np.median(flux_padded[i: i+window_size])

    return flux_filtered


#### sigma clipping #### 
def sigma_clip(time,flux,sigma):
    time = np.asarray(time)
    flux = np.asarray(flux)
    mask = np.ones(len(flux),dtype=bool)
    while True:
        mean_flux = np.mean(flux[mask])
        std_flux = np.std(flux[mask])
        new_mask = np.abs(flux-mean_flux)<sigma*std_flux
        if np.array_equal(mask,new_mask):
            break
        mask = new_mask
    return time[mask], flux[mask]


#########################################################################
# Code examples from Mikkel (parameters):
#########################################################################

def logmed_filter(frequency,power,filter_width=0.01):
    
    count = np.zeros(len(frequency),dtype=int)
    bkg = np.zeros_like(frequency)
    x0 = np.log10(frequency[0])
    logf = np.log10(frequency)
    corr_factor = (8.0 / 9.0)**3
    while x0 < np.log10(frequency[-1]):
        m = (np.abs(logf - x0) < filter_width)
        if len(bkg[m]) > 0:
            bkg[m] += nanmedian(power[m]) / corr_factor
            count[m] += 1
            x0 += 0.5 * filter_width
        else:
            print('Choose a larger filter with')
            break
    
    smooth_pg = bkg / count
    return smooth_pg

#### Autocorrelation #### 
def autocorr_fft(x):
    xp = x-np.nanmean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:int(x.size/2)]/np.sum(xp**2)

def autocorr(x):
    X = x- np.nanmean(x)
    result = np.correlate(X,X,mode='same')
    return result[int(result.size/2):]

#### Finding maximum of peaks #### 
def find_maximum(guess,X,Y,minx,maxx):
    fx = INT(X,Y)
    def opt(x): return -fx(x)
    opt = OP.minimize(opt, guess,method='Nelder-Mead', bounds=[(minx,maxx)])
    return opt.x

#### 
def dnu_guess(numax):
    alpha_u, beta_u = 0.267, 0.760
    alpha_o, beta_o = 0.22, 0.797

    Dnu_guess = alpha_u * ( numax<300)*numax**beta_u
    Dnu_guess += alpha_o * ( numax>=300)*numax**beta_o
    return Dnu_guess
    

#### Making Echelle diagram #### 
def Echellediagram(Power, Fre,lsep,Start,Stop,no_repeat=1):

    sta = np.argmin(np.abs(Fre-Start))
    sto = np.argmin(np.abs(Fre -Stop))
    
    step = np.median(np.diff(Fre))
    Lsep_len = int(round(lsep/step))
    Lsep_len2 = int(round(lsep/step))*no_repeat
    N = int(np.ceil((len(Fre[sta:sto])/Lsep_len)))

    ValsP = np.zeros([N,Lsep_len2])
    FreVec_y = np.zeros(N)
    for i in np.arange(N):
        if i==0:
            start = sta
            stop = sta + Lsep_len2
            FreVec_y[i] = lsep/2 + Start
            FreVec_x = np.mod(Fre[start:stop], lsep*no_repeat)
            idx_sort = np.argsort(FreVec_x)
            FreVec_y[i] = FreVec_y[i-1] + lsep
        else:
            start = stop - Lsep_len*(no_repeat-1)
            stop = int(start + Lsep_len2)
            FreVec_y[i] = FreVec_y[i-1] + lsep

        ValsP[i,:] = Power[start:stop][idx_sort]

    EchelleVals = ValsP[::-1]
    return FreVec_x, FreVec_y, EchelleVals


def crange2(start, stop, vals):
    vals_to_return = np.array([])
    for val in vals:
        if val>stop:
            vals_to_return = np.append(vals_to_return,
                                       np.mod(val,stop) + start)
        else:
            vals_to_return = np.append(vals_to_return, val)
    return vals_to_return


def OAPS(f, p ,start, stop, dnus, K=4):
    '''
    Function that computes the order averaged power spectrum
    '''
    start_idx = np.argmin(np.abs(f-start))
    stop_idx = np.argmin(np.abs(f-stop))
    #K deifnes the FWHM of weights as K*dnu
    CC = 0
    for j in range(-K,K+1,1):
        c = 1/(1 + (2*np.abs(j)/K)**2)
        CC+=c

    #pre allocating space
    new_fre = np.zeros([len(dnus), stop_idx-start_idx+1])
    oaps = np.zeros([len(dnus), stop_idx-start_idx+1])

    #run order averaing (vertical echelle smoothing)
    #run a range of dnu values to capture curvature
    for k, dnu in enumerate(dnus):
        print(k, 'running OAPS with dnu ', dnu)
        idx = start_idx.copy()
        for i in range(new_fre.shape[1]):

            OAP = 0
            for j in range(-K, K+1, 1):
                c = 1 / (1 + (2*np.abs(j)/K)**2)
                idx_fre = np.argmin(np.abs(f - (f[idx+i] + j*dnu)))
                OAP += (c/CC) * p[idx_fre]
            oaps[k,i] = OAP
            new_fre[k, i] = f[idx+i]
    #final spectrum is for each frequency the OAPS for the dnu,
    # with the highest response
    av_oaps = np.max(oaps, axis=0)
    oaps_fre = new_fre[0,:]
    return oaps_fre, av_oaps, oaps
                

    
#########################################################################
# Other functions:
#########################################################################


def scaling_relations(numax,dnu,Teff):
    #Chaplin et al. 2014:
    numax_sun = 3090 #muHz
    dnu_sun = 135.1 #muHz
    #Values in solar masses:
    M = (numax / numax_sun)**3 * (dnu_dnu_sun)**(-4) * (Teff/Teff_sun)**(3/2)
    R = (numax / numax_sun) * (dnu_dnu_sun)**(-2) * (Teff/Teff_sun)**(1/2)
    return M, R

def dnu_from_numax(numax):
    if numax < 300: #muHz
        alpha, beta = 0.259, 0.765
    else:
        alpha, beta = 0.25, 0.779
    return alpha * (numax)**beta
        

















