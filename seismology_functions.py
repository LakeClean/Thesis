import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ps import powerspectrum
from scipy.interpolate import InterpolatedUnivariateSpline as INT
from scipy import optimize as OP
import glob
from ophobningslov import *
from scipy.optimize import minimize
import lmfit
from scipy.ndimage import gaussian_filter
import emcee
import os
from multiprocessing import Pool
import corner
import time


os.environ["OMP_NUM_THREADS"] = "1" #recommended setting for parallelizing emcee

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


def scaling_relations(numax, e_numax,dnu,e_dnu,Teff):
    #Chaplin et al. 2014:
    numax_sun = 3090 #muHz
    dnu_sun = 135.1 #muHz
    Teff_sun = 5780 #K
    #Values in solar masses:
    varsAndVals = {'numax':[numax,e_numax],'dnu':[dnu,e_dnu]}
    M = (numax / numax_sun)**3 * (dnu/dnu_sun)**(-4) * (Teff/Teff_sun)**(3/2)
    R = (numax / numax_sun) * (dnu/dnu_sun)**(-2) * (Teff/Teff_sun)**(1/2)
    unc_M = f'(numax / {numax_sun})**3 * (dnu/{dnu_sun})**(-4) * ({Teff}/{Teff_sun})**(3/2)'
    unc_R = f'(numax / {numax_sun}) * (dnu/{dnu_sun})**(-2) * ({Teff/Teff_sun})**(1/2)'
    e_M = ophobning(unc_M,varsAndVals,False)
    e_R = ophobning(unc_R,varsAndVals,False)
    return M, R, e_M, e_R


def dnu_from_numax(numax):
    if numax < 300: #muHz
        alpha, beta = 0.259, 0.765
    else:
        alpha, beta = 0.25, 0.779
    return alpha * (numax)**beta


def background(x, theta):
    out = 0
    for i in range(int(len(theta)/2)):
        i *=2
        Ai,Bi = theta[i], theta[i+1]
        out += Ai / (1 + (Bi*x)**2)
    return out

def background_res(params,x,y):
    values = []
    for i in list(params.values()):
        values.append(i.value)
    res = y - background(x,values)
    return res


def Gaussian(x,a,b,c,floor):
    return a*np.exp(-0.5*((x-b)/c)**2) + floor

def Gaussian_res(params,x,y,weight=1):
    '''
    Returns the residual between Gaussian and data weighted by the error.
    '''
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    floor = params['floor'].value
    res = y - Gaussian(x,a,b,c,floor)
    return res * weight


def mode(x,theta):
    '''
    Model of a single peak in the power spectrum
    '''
    epsH,gam,nu,const = theta
    return epsH / (1 + 4/gam**2 * (x - nu)**2) + const

def mode_res(params,x,y):
        epsH = params['epsH'].value
        gam = params['gam'].value
        nu = params['nu'].value
        const = params['const'].value
        res = y - mode(x,[epsH,gam,nu,const])
        return res

def mode_double(x,theta):
    '''
    Model for two peaks in the power spectrum
    '''
    epsH1,gam1,nu1,const,epsH2,gam2,nu2 = theta
    out = epsH1 / (1 + 4/gam1**2 * (x - nu1)**2) + const
    out += epsH2 / (1 + 4/gam2**2 * (x - nu2)**2) + const
    return out

def mode_double_res(params,x,y):
    epsH1 = params['epsH1'].value

    gam1 = params['gam1'].value
    nu1 = params['nu1'].value
    const = params['const'].value
    epsH2 = params['epsH2'].value
    gam2 = params['gam2'].value
    nu2 = params['nu2'].value
    res = y - mode_double(x,[epsH1,gam1,nu1,const,epsH2,gam2,nu2])
    return res



def log_likelihood(theta,xs,ys):
    out = 0
    for x,y in zip(xs,ys):
        out -= np.log(mode(x,theta)) + y/mode(x,theta)
    return out


def log_prior(theta):
    '''
    input:
        -theta: list of parameters given to model (mode)
        -prior: list of lists with "limits" of every parameter
    return:
        If value lies within priors then return 0. If not return neg inf.
    '''
    epsH,gam,nu,const = theta
    if 0 < epsH < 1000 and -1000 < gam < 1000 and 0 < nu < 10000 and 0<const<1000:
        return 0.0
    return -np.inf

def log_probability(theta,x,y):
    '''
    Given the parameters (theta) provided the prior is evaluated.
    The result is either 0 or -inf. We check if it is finite.
    If that is the case we return the ln_likelihood function given the
    parameters theta that we checked against the priors, and the dataset.
    This will then be a probability of taking the step maybe?
    If it is not finite, i.e. the parameters theta lie outside the priors,
    then we return -inf. The step is then not taken??? idk.
    '''
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x,y)

def log_likelihood_double(theta,xs,ys):
        out = 0
        for x,y in zip(xs,ys):
            out -= np.log(mode_double(x,theta)) + y/mode_double(x,theta)
        return out

def log_prior_double(theta):
    epsH1,gam1,nu1,const,epsH2,gam2,nu2 = theta
    if 0 < epsH1 < 1000 and -1000 < gam1 < 1000 and 0 < nu1 < 10000 and 0<const<1000:
        if 0 < epsH2 < 1000 and -1000 < gam2 < 1000 and 0 < nu2 < 10000:
            return 0.0
    return -np.inf

def log_probability_double(theta,x,y):
    lp = log_prior_double(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_double(theta, x,y)




def mode_N(x,theta):
    '''
    Model for N peaks in the power spectrum and a constant background
    '''
    const = theta[-1]#We keep gammma and constant offset same for each peak
    gam = theta[-2] 
    out = 0
    for i in range(int(len(theta)/2) - 1):
        i *=2
        epsH, nu = theta[i], theta[i+1]
        out += epsH / (1 + 4/gam**2 * (x - nu)**2) + const
    return out

def mode_N_res(params,x,y):
    #print(list(params.values())[0].name)
    values = []
    for i in list(params.values()):
        values.append(i.value)
        
    res = y - mode_N(x,values)
    return res


def log_likelihood_N(theta,xs,ys):
        out = 0
        for x,y in zip(xs,ys):
            out -= np.log(mode_N(x,theta)) + y/mode_N(x,theta)
        return out


def log_prior_N(theta, bounds):
    const = theta[-1]
    gam = theta[-2]
    if not bounds[-2][0] < gam < bounds[-2][1] and bounds[-1][0]<const<bounds[-1][1]:
        return -np.inf
    out = 0.0
    for i in range(int(len(theta)/2) - 1):
        i *= 2
        epsH, nu = theta[i], theta[i+1]
        if not bounds[i][0] < epsH < bounds[i][1] and bounds[i+1][0] < nu < bounds[i+1][1]:
            return -np.inf
    return out

def log_probability_N(theta,x,y,bounds):
    lp = log_prior_N(theta,bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_N(theta, x,y)

def int_of_peak(lim_a,lim_b,params,e_params=[]):
    '''
    Compute the analytical integral of the peak.
    params: list of params
    e_params: list of errors in params
    lim_a: the lower limit on integral
    '''
    epsH,nu,gam,const = params

    end = -epsH * np.arctan(2/gam * (nu - lim_a)) / (2/gam)
    start = -epsH * np.arctan(2/gam * (nu - lim_b)) / (2/gam)
    integral = start - end

    if e_params == []:
        return integral
    
    e_epsH,e_gam,e_nu,e_const = e_params

    varsAndvals = {'epsH': [epsH,e_epsH], 'gam':[gam,e_gam],
                   'nu': [nu,e_nu], 'const':[const,e_const]}

    e_end = f'-epsH * atan(2/gam * (nu - {lim_a})) / (2/gam)'
    e_start = f'-epsH * atan(2/gam * (nu - {lim_b})) / (2/gam)'
    e_integral = ophobning(f'{e_start} - {e_end}',varsAndvals,False)

    
    
    return integral, e_integral


#########################################################################
# Testing:
#########################################################################
'''
#importing fitting parameters
master_path = '/usr/users/au662080'
ID = 'KIC10454113'
mode_params = [[],[],[]]
guess_points = [[],[],[]]
ind_peaks_path = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'

for k in range(3):
    mode_path = ind_peaks_path+f'individual_peaks_mode{k+1}.txt'
    ind_peaks_mode_df = pd.read_csv(mode_path).to_numpy()
    mode_params[k].append(ind_peaks_mode_df)

    eye_path = ind_peaks_path+f'individual_peaks_eye{k+1}.txt'
    ind_peaks_eye_df = pd.read_csv(eye_path).to_numpy()
    guess_points[k].append(ind_peaks_eye_df)
        


#importing frequency and power
power_path = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
power_path += 'power_spec.txt'

power_df = pd.read_csv(power_path)
freq = power_df['Frequency'].to_numpy()
power = power_df['power'].to_numpy()



fig, ax = plt.subplots()

ax.plot(freq,power)
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()



reg1 = 20 #before
reg2 = 50 #inbetween
reg3 = 20 #after
for i in range(len(mode_params[0][0])):
    
    nr_peaks = 0 #int, whether there is a peak very close

    z_point = guess_points[0][0][i][0]
    guess1 = guess_points[0][0][i][0]

    points = [z_point]
    #Finding out if there is a point(s) close to the peak we are looking at
    for j in range(len(mode_params[1][0])):
        x_point = guess_points[1][0][j][0]

        if abs(x_point - z_point) < reg2:
            nr_peaks = 1
            x_idx = j
            points.append(x_point)
            for k in range(len(mode_params[2][0])):
                v_point = guess_points[2][0][k][0]
                
                if abs(x_point - v_point) < reg1:
                    nr_peaks = 2
                    v_idx = k
                    points.append(v_point)
                    break
            break
                
                

    if nr_peaks:
        print(f'{nr_peaks} peaks close!')

    fig, ax = plt.subplots()
    
    #isolating the part of the spectrum we want to fit
    if nr_peaks == 0:
        f_idx = (( z_point - reg1 < freq ) &
                 ( z_point + reg3 > freq ) )
    if nr_peaks == 1:
        f_idx = (( x_point - reg1 < freq ) &
                 ( z_point + reg3 > freq ) )
        
    if nr_peaks == 2:
        f_idx = (( v_point - reg1 < freq ) &
                 ( z_point + reg3 > freq ) )
        
        
    f = freq[f_idx]
    p = power[f_idx]
    

    #fitting with least squares

    start_time = time.time()
    params = lmfit.Parameters()

    for j in range(nr_peaks+1):
        params.add(f'epsH{j}',value=3)
        params.add(f'nu{j}',points[j])

    params.add(f'gam',value=5)
    params.add('const',value=0)

    fit = lmfit.minimize(mode_N_res, params, args=(f,p),
                         xtol=1.e-7,ftol=1.e-7,max_nfev=500)
    
    print(lmfit.fit_report(fit,show_correl=False))

    values = []
    for i in list(fit.params.values()):
        values.append(i.value)


    ax.plot(f,mode_N(f,values),
            label='least squares fit', color='red')
    print('Time to least squares fit ', time.time() - start_time)

    #Minimizing:
    start_time = time.time()
    print(values)
    nll = lambda *args: -log_likelihood_N(*args)
    initial = np.array(values)

    bounds = []
    for j in range(nr_peaks+1):
        j *= 2
        bounds.append((0.000001,10))#epshH
        bounds.append((initial[j+1]-100,initial[j+1]+100))#nu
    bounds.append((0.000001,10))# gam
    bounds.append((0.000001,10))#const
    
    soln = minimize(nll, initial, args=(f,p),
                    bounds=bounds,
                    method = 'Nelder-Mead')
    print(soln.x)
    print(soln)
    print('Time to max likelihood ', time.time() - start_time)

    #Smothing to illustrate fit
    smoothed = gaussian_filter(p, sigma=6)

    #plotting

    #ax.plot(freq,power)
    ax.plot(f,p,label='power spectrum',zorder=1)

    ax.plot(f, mode_N(f,soln.x),label='maximized likelihood')

    ax.plot(f,smoothed,label='smoothed',zorder=1)

    ax.legend()

    
    plt.show()

    #trying emcee:



    nwalkers,ndim = 20,len(values)
    iterations = 800

    pos = np.ones(shape=(nwalkers,ndim))#allocating space
    for j,val in enumerate(soln.x):
        pos[:,j] = abs(np.random.normal(loc = val, scale = 0.1, size=nwalkers))

    #pos = soln.x + 1e-4*np.random.normal(nwalkers,ndim)
    
    with Pool(7) as pool:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                        log_probability_N,args=(f,p,bounds),
                                        pool=pool)
        sampler.run_mcmc(pos,iterations,progress=True)

    fig, ax = plt.subplots(ndim, figsize=(10, ndim), sharex=True)
    samples = sampler.get_chain()
    labels = []
    for i in list(fit.params.values()):
        labels.append(i.name)
        
    for i in range(ndim):
        ax[i].plot(samples[:, :, i], "k", alpha=0.3)
        ax[i].set_xlim(0, len(samples))
        ax[i].set_ylabel(labels[i])
        ax[i].yaxis.set_label_coords(-0.1, 0.5)

    ax[-1].set_xlabel("step number")

    plt.show()

    fig,ax = plt.subplots()

    flat_samples = sampler.get_chain(discard=100, flat=True)

    corner.corner(flat_samples, labels=labels,
                        truths=soln.x)

    plt.show()

'''






'''
mode_nr = 1
reg = 20
for i in range(len(mode_params[mode_nr][0])):
    
    close = False #bool, whether there is a peak very close

    x_point = guess_points[mode_nr][0][i][0]
    #eps1, H1, gam1, nu1,const1 = mode_params[mode_nr][0][i]
    guess1 = guess_points[mode_nr][0][i][0]
    
    if mode_nr == 1:
        for j in range(len(mode_params[2][0])):
            v_point = guess_points[2][0][j][0]

            #eps2, H2, gam2, nu2, const2 = mode_params[2][0][j]
            
            if abs(x_point - v_point) < 20:
                close = True
                break

    if close:
        print('close!')

    fig, ax = plt.subplots()
    
    #isolating the part of the spectrum we want to fit
    x_idx = (( x_point - reg < freq ) &
             ( x_point + reg > freq ) )
    f = freq[x_idx]
    p = power[x_idx]
    



    #fitting with least squares

    if close:
        params = lmfit.Parameters()
        params.add('epsH1',value=3)
        params.add('gam1',value=10)
        params.add('nu1',x_point)
        params.add('const',value=0)
        params.add('epsH2',value=3)
        params.add('gam2',value=10)
        params.add('nu2',v_point)
        

        fit = lmfit.minimize(mode_double_res, params, args=(f,p),
                             xtol=1.e-8,ftol=1.e-8,max_nfev=500)
        
        print(lmfit.fit_report(fit,show_correl=False))

        epsH1 = fit.params['epsH1'].value
        gam1 = fit.params['gam1'].value
        nu1 = fit.params['nu1'].value
        const = fit.params['const'].value
        epsH2 = fit.params['epsH2'].value
        gam2 = fit.params['gam2'].value
        nu2 = fit.params['nu2'].value

        ax.plot(f,mode_double(f,[epsH1,gam1,nu1,const,epsH2,gam2,nu2]),
                label='least squares fit', color='red')
    else:
        params = lmfit.Parameters()
        params.add('epsH',value=3)
        params.add('gam',value=10)
        params.add('nu',x_point)
        params.add('const',value=0)

        fit = lmfit.minimize(mode_res, params, args=(f,p),
                             xtol=1.e-8,ftol=1.e-8,max_nfev=500)
        
        print(lmfit.fit_report(fit,show_correl=False))

        epsH = fit.params['epsH'].value
        gam = fit.params['gam'].value
        nu = fit.params['nu'].value
        const = fit.params['const'].value

        ax.plot(f,mode(f,[epsH,gam,nu,const]),label='least squares fit', color='red')
    


    #Minimizing:

    if close:
        nll = lambda *args: -log_likelihood_double(*args)
        initial = np.array([epsH1, gam1, nu1,const,epsH2, gam2, nu2])

        soln = minimize(nll, initial, args=(f,p),
                        bounds=[(0.000001,10),(-10,10),
                                (0.000001,10000),(0.000001,20),
                                (0.000001,10),
                                (-10,10),(0.000001,10000)],
                        method = 'Nelder-Mead')
        print(soln.x)
        print(soln)


    else:

        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([epsH, gam, nu,const])

        soln = minimize(nll, initial, args=(f,p),
                        bounds=[(0.000001,10),(-10,10),
                                (0.000001,10000),(0.000001,20)],
                        method = 'Nelder-Mead')
        print(soln.x)
        print(soln)



    #Smothing to illustrate 
    smoothed = gaussian_filter(p, sigma=6)

    #plotting
    
    ax.plot(f,p,label='power spectrum',zorder=1)

    if close:
        ax.plot(f, mode_double(f,soln.x),label='maximized likelihood')
    else:
        ax.plot(f, mode(f,soln.x),label='maximized likelihood')
    ax.plot(f,smoothed,label='smoothed')

    ax.legend()

    
    plt.show()


    #trying emcee:


    if close:
        nwalkers,ndim = 15,7
        iterations = 1000
        pos = soln.x + 1e-4*np.random.randn(nwalkers,ndim)
        
        with Pool(6) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                            log_probability_double,args=(f,p),
                                            pool=pool)
            sampler.run_mcmc(pos,iterations,progress=True)

        fig, ax = plt.subplots(ndim, figsize=(10, ndim), sharex=True)
        samples = sampler.get_chain()
        labels = ["epsH1", "gam1", "nu1", "const","epsH2", "gam2", "nu2"]
        for i in range(ndim):
            ax[i].plot(samples[:, :, i], "k", alpha=0.3)
            ax[i].set_xlim(0, len(samples))
            ax[i].set_ylabel(labels[i])
            ax[i].yaxis.set_label_coords(-0.1, 0.5)

        ax[-1].set_xlabel("step number")

        plt.show()

        fig,ax = plt.subplots()

        flat_samples = sampler.get_chain(discard=100, flat=True)

        corner.corner(flat_samples, labels=labels,
                            truths=soln.x)

        plt.show()

    else:
        nwalkers,ndim = 15,4
        
        iterations = 1000
        pos = soln.x + 1e-4*np.random.randn(nwalkers,ndim)

        
        with Pool(6) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                            log_probability,args=(f,p),
                                            pool=pool)
            sampler.run_mcmc(pos,iterations,progress=True)

        fig, ax = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["epsH", "gam", "nu", "const"]
        for i in range(ndim):
            ax[i].plot(samples[:, :, i], "k", alpha=0.3)
            ax[i].set_xlim(0, len(samples))
            ax[i].set_ylabel(labels[i])
            ax[i].yaxis.set_label_coords(-0.1, 0.5)

        ax[-1].set_xlabel("step number")

        plt.show()

        fig,ax = plt.subplots()

        flat_samples = sampler.get_chain(discard=100, flat=True)

        corner.corner(flat_samples, labels=['epsH', 'gam', 'nu', 'const'],
                            truths=soln.x)

        plt.show()
   


    
'''
















