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
    epsH,gam,nu,const = theta
    return epsH / (1 + 4/gam**2 * (x - nu)**2) + const

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
    if 0 < epsH < 1000 and 0 < gam < 1000 and 0 < nu < 10000 and 0<const<1000:
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



def int_of_peak(lim_a,lim_b,params,e_params):
    '''
    Compute the analytical integral of the peak.
    params: list of params
    e_params: list of errors in params
    lim_a: the lower limit on integral
    '''
    epsH,gam,nu,const = params
    e_epsH,e_gam,e_nu,e_const = e_params

    varsAndvals = {'epsH': [epsH,e_epsH], 'gam':[gam,e_gam],
                   'nu': [nu,e_nu], 'const':[const,e_const]}

    
    e_end = f'-epsH * atan(sqrt(4)/gam * (nu - {lim_a})) / (sqrt(4)/gam) + const*{lim_a}'
    e_start = f'-epsH * atan(sqrt(4)/gam * (nu - {lim_b})) / (sqrt(4)/gam) + const*{lim_b}'
    e_integral = ophobning(f'{e_start} - {e_end}',varsAndvals,False)

    end = -epsH * np.arctan(np.sqrt(4)/gam * (nu - lim_a)) / (np.sqrt(4)/gam) + const*lim_a
    start = -epsH * np.arctan(np.sqrt(4)/gam * (nu - lim_b)) / (np.sqrt(4)/gam) + const*lim_b
    integral = start - end
    
    return integral, e_integral

    
    


##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True):
    '''
    Function for finding numax and deltanu of given target.
    Also does a lot of plotting.
    Give data as power spectrum. and allow only for target with single visible
    seismic signal.
    '''
    
    ##################### Importing info: ###########################
    ID_idx = np.where(ID == IDs)[0]
    if len(ID_idx) != 1:
        print('ID was given wrong')
        print('The ID should be among the following:')
        print(IDs)
        return 0

    data_type = data_types[ID_idx][0]
    numax_guess = numax_guesss[ID_idx][0]
    data_path = data_paths[ID_idx][0]
    print('Analysing ',ID, '|  numax guess:', numax_guess)


    #Power spec unfiltered
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    power_spec_df = pd.read_csv(path_to_save + 'power_spec.txt')



    

    
    #################### Analysing: ################################

    
    if find_ind_peaks:
        guess_points, gauss_params, mode_params, region = simple_peak_bagging(f[idx1],
                                                                p[idx1])
        if saving_data:
            path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
            np.save(path_to_save + 'individual_peaks_mode',mode_params)
            np.save(path_to_save + 'region',region)
            np.save(path_to_save + 'individual_peaks_eye',guess_points)
            
            

        
        #maximum likelihood
        ml_params = np.zeros(shape=(mode_params.shape[0],mode_params.shape[1],4))
        #maximum likelihood error from MCMC
        e_ml_params = np.zeros(shape=(mode_params.shape[0],mode_params.shape[1],4))

        amplitudes = np.zeros(shape = (ml_params.shape[0],ml_params.shape[1]))
        e_amplitudes = np.zeros(shape = (ml_params.shape[0],ml_params.shape[1]))


        #logfile for fitting:
        fitlogfile = open(f'fitting_log_{ID}.txt','w')
        fitlogfile.write('The fits of the following peaks failed\n')
        

        
        for k in range(len(guess_points)):
            for i, point in enumerate(guess_points[k]):
                if point[0] == 0: #checking if point exists
                    continue

                eps_guess,H_guess,gam_guess,nu_guess,const_guess = mode_params[k,i,:]
                

                #limiting power spectrum to around peak:
                idx_peak = (point[0] - region[k]<f) & (point[0] + region[k]>f)

                #First we estimate parameters with maximum likelihood
                nll = lambda *args: -log_likelihood(*args)
                initial = np.array([eps_guess*H_guess,gam_guess,nu_guess,const_guess])

                soln = minimize(nll, initial, args=(f[idx_peak],p[idx_peak]),
                                bounds=[(0.000001,10),(-100,100),
                                        (0.000001,10000),(0.000001,20)],
                                method = 'Nelder-Mead')
                for init,bound in zip(initial,[(0.000001,10),(-100,100),
                                        (0.000001,10000),(0.000001,20)]):
                    if (init > bound[1]) or (init < bound[0] ):
                        print(initial)
                        fitlogfile.write(f'{point[0]}\n')
                        
                        
                        
                if False:
                    fig,ax = plt.subplots()
                    ax.plot(f[idx_peak],p[idx_peak])
                    ax.plot(f[idx_peak],mode(f[idx_peak],soln.x),label='max_likelihood')
                    ax.plot(f[idx_peak],
                            mode(f[idx_peak],[eps_guess*H_guess,gam_guess,nu_guess,const_guess]),
                            label='least squares')
                    ax.legend()
                    plt.show()
                
                #We find uncertainties of parameters with emcee
                nwalkers,ndim = 15,4
                iterations = 1000
                burnin = 100
                pos = soln.x + 1e-4*np.random.randn(nwalkers,ndim)

                
                if True: #if using multiprocessing
                    with Pool(5) as pool:
                        sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                                        log_probability,args=(f[idx_peak],p[idx_peak]),
                                                        pool=pool)
                        sampler.run_mcmc(pos,iterations,progress=True)
                else:
                    sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                                        log_probability,
                                                    args=(f[idx_peak],p[idx_peak]))
                    sampler.run_mcmc(pos,iterations,progress=True)
                    
                    
                samples = sampler.get_chain()
                labels = ["epsH", "gam", "nu", "const"]
                flat_samples = sampler.get_chain(discard=burnin, flat=True)

                params = []
                e_params = []
                for j in range(ndim):
                    percentiles = np.percentile(flat_samples[:,j], [16,50,84])
                    q = np.diff(percentiles)
                    params.append(percentiles[1])
                    e_params.append(max(q))


                #plotting sampling
                if False:
                    fig, ax = plt.subplots(4, figsize=(10, 7), sharex=True)
                    samples = sampler.get_chain()
                    labels = ["epsH", "gam", "nu", "const"]
                    for i in range(ndim):
                        ax[i].plot(samples[:, :, i], "k", alpha=0.3)
                        ax[i].set_xlim(0, len(samples))
                        ax[i].set_ylabel(labels[i])
                        ax[i].yaxis.set_label_coords(-0.1, 0.5)
                    plt.show()

                #plotting corner plot
                if False:
                    fig,ax = plt.subplots()
                    flat_samples = sampler.get_chain(discard=100, flat=True)
                    corner.corner(flat_samples, labels=['epsH', 'gam', 'nu', 'const'],truths=soln.x)
                    plt.show()



                #Finding amplitude of peak
                lim_a = point[0] - region[k]
                lim_b = point[0] + region[k]
                
                integral,e_integral = int_of_peak(lim_a,lim_b, params,e_params)
                    
                amplitudes[k,i] = integral
                e_amplitudes[k,i] = e_integral

                ml_params[k,i,:] = params
                e_ml_params[k,i,:] = e_params

        fitlogfile.close()


        if saving_data:
            path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
            np.save(path_to_save + 'individual_peaks_max_like',ml_params)
            np.save(path_to_save + 'individual_peaks_e_max_like',e_ml_params)
            np.save(path_to_save + 'individual_peaks_amplitude',amplitudes)
            np.save(path_to_save + 'individual_peaks_e_amplitude',e_amplitudes)
        
    else:

        
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        ml_params = np.load(path_to_save + 'individual_peaks_max_like.npy')
        e_ml_params = np.load(path_to_save + 'individual_peaks_e_max_like.npy')
        guess_points = np.load(path_to_save + 'individual_peaks_eye.npy')
        amplitudes = np.load(path_to_save + 'individual_peaks_amplitude.npy')
        e_amplitudes = np.load(path_to_save + 'individual_peaks_e_amplitude.npy')
            
    
    #Finding numax:
    alt_dnus = np.zeros(3) #an alternative dnu estimate
    e_alt_dnus = np.zeros(3) # error in alternative dnu estimate
    numax_est = np.zeros(3) #positions of gaussian for modes
    e_numax_est = np.zeros(3) #'error' of gaussian for modes

    
    
    fig, ax = plt.subplots()
    for k in range(len(ml_params)):
        

        #sorting peaks:
        sorted_amplitudes = [x for y,x in zip(ml_params[k,:,2],amplitudes[k]) if y!=0]
        sorted_e_amplitudes = [x for y,x in zip(ml_params[k,:,2],e_amplitudes[k]) if y!=0]
        sorted_frequency_peak = [x for x in ml_params[k,:,2] if x!=0 ]

        
        

        #finding alternative dnu
        peak_diffs = np.diff(sorted_frequency_peak)
        alt_dnu = np.mean(peak_diffs)
        e_alt_dnu = np.std(peak_diffs)/np.sqrt(len(peak_diffs))
        print('alternative dnu:',alt_dnu,'muHz+/-', e_alt_dnu)

        if len(sorted_amplitudes) <4:
            continue

        
        #Fitting Gaussian
        params = lmfit.Parameters()
        params.add('a',value=max(sorted_amplitudes))
        params.add('b', value=numax_guess)
        params.add('c', value=np.std(sorted_frequency_peak))
        params.add('floor',value=0)

        fit = lmfit.minimize(Gaussian_res, params,
                             args=(sorted_frequency_peak,sorted_amplitudes),
                             xtol=1.e-8,ftol=1.e-8,max_nfev=500)
        print(lmfit.fit_report(fit,show_correl=False))
        a = fit.params['a'].value
        b = fit.params['b'].value
        c = fit.params['c'].value
        floor = fit.params['floor'].value

        print('covariance:',fit.covar)
        try:
            e_a,e_b,e_c,e_floor = np.diagonal(fit.covar)
        except:
            e_a,e_b,e_c,e_floor = 0,0,0,0

        
        numax_est[k] = b
        e_numax_est[k] = e_b

        

        freq_space = np.linspace(min(sorted_frequency_peak),max(sorted_frequency_peak),100)
        norm = max(Gaussian(freq_space,a,b,c,floor))
        
        ax.plot(freq_space,Gaussian(freq_space,a,b,c,floor),ls='--',
                                  zorder=3)
        ax.fill_between(freq_space,Gaussian(freq_space,a-e_a,b-e_b,c-e_c,floor-e_floor),
                        Gaussian(freq_space,a+e_a,b+e_b,c+e_c,floor+e_floor),alpha=0.3,zorder=1)

        ax.scatter(sorted_frequency_peak,sorted_amplitudes)
        
        ax.set_title(f'{ID}')
        ax.set_xlabel('Center freuqency')
        ax.set_ylabel('Amplitude [area under peak]')
    plt.show()

    if saving_data: save_data(ID, 'numax',
                                  [numax_est,e_numax_est],
                                  ['numax','numax_error'])
    
    if saving_data: save_data(ID, 'alt_dnu', [alt_dnus,e_alt_dnus],
                                  ['alt_dnu', 'e_alt_dnu'])





if False:
    analyse_power('KIC10454113',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True) 

if False:  
    analyse_power('KIC9025370',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True)

if False: 
    analyse_power('KIC12317678',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = False)

if True: 
    analyse_power('KIC4914923',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True)

'''
if True:
    analyse_power('EPIC236224056',saving_data = True, plotting = False,
                  filtering = True,find_ind_peaks = False)
if True:
    analyse_power('EPIC246696804',saving_data = True, plotting = False,
                  filtering = True,find_ind_peaks = False)
if True:
    analyse_power('EPIC249570007',saving_data = True, plotting = False,
                  filtering = True,find_ind_peaks = False)
if True:
    analyse_power('EPIC230748783',saving_data = True, plotting = False,
                  filtering = True,find_ind_peaks = False)
if True:
    analyse_power('EPIC212617037',saving_data = True, plotting = False,
                  filtering = True,find_ind_peaks = False)
'''











