import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from ps import powerspectrum
#from bottleneck import nanmedian, nanmean, nanmax, nanmin
#from scipy.interpolate import InterpolatedUnivariateSpline as INT
#pofrom scipy import optimize as OP
#from matplotlib.colors import LogNorm
#from matplotlib import cm
import glob
import seismology_functions as sf
#from matplotlib.widgets import Button, Slider
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
import os

os.environ["OMP_NUM_THREADS"] = "1" #recommended setting for parallelizing emcee
master_path = '/usr/users/au662080'


#importing log file
log_file_path = f'{master_path}/Speciale/data/Seismology/analysis/'
log_file_path += 'log_file.txt'
log_df = pd.read_csv(log_file_path)
IDs = log_df['ID'].to_numpy()




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

def mode_res(params,x,y):
        epsH = params['epsH'].value
        gam = params['gam'].value
        nu = params['nu'].value
        const = params['const'].value
        res = y - mode(x,[epsH,gam,nu,const])
        return res

def mode_double(x,theta):
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

    
    e_end = f'-epsH * atan(2/gam * (nu - {lim_a})) / (2/gam)'
    e_start = f'-epsH * atan(2/gam * (nu - {lim_b})) / (2/gam)'
    e_integral = ophobning(f'{e_start} - {e_end}',varsAndvals,False)

    end = -epsH * np.arctan(2/gam * (nu - lim_a)) / (2/gam)
    start = -epsH * np.arctan(2/gam * (nu - lim_b)) / (2/gam)
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


    #Power spec unfiltered
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    power_spec_df = pd.read_csv(path_to_save + 'power_spec.txt').to_numpy()
    f,p = power_spec_df[:,0], power_spec_df[:,1]

    #importing peakinfo
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    guess_points = np.load( path_to_save + 'individual_peaks_eye.npy')

    ml_params = np.load(path_to_save + 'individual_peaks_max_like.npy')
    amplitudes = np.zeros(shape=(ml_params.shape[0],ml_params.shape[1]))
    e_amplitudes = np.zeros(shape=(ml_params.shape[0],ml_params.shape[1]))
    
        
    ################## Errors through MCMC fitting ################################
    reg = 20
    for k in range(len(guess_points) -1 ):
        for i, point in enumerate(guess_points[k]):
            close = False #bool, whether there is a peak very close
            x_point = point[0]

            #checking if point exists
            if x_point == 0: 
                continue

            if k == 1:
                for j in range(len(mode_params[2][0])):
                    v_point = guess_points[2][j][0]
                    epsH2, gam2, nu2, const2 = ml_params[2,j,:]
                    
                    if abs(x_point - v_point) < 20:
                        close = True
                        weak_idx = j
                        break

            epsH1, gam1, nu1, const1 = ml_params[k,i,:]
            #limiting power spectrum to around peak:
            idx_peak = (point[0] - reg<f) & (point[0] + reg>f)

            if close:
                nwalkers,ndim = 15,7
                iterations = 1000
                params = np.array([epsH1,gam1,nu1,const1,epsH2,gam2,nu2])
                pos = params + 1e-4*np.random.randn(nwalkers,ndim)
                
                with Pool(6) as pool:
                    sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                                    log_probability_double,
                                                    args=(f[idx_peak],p[idx_peak]),
                                                    pool=pool)
                    sampler.run_mcmc(pos,iterations,progress=True)


                #plotting sampling
                if plotting:
                    fig, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
                    samples = sampler.get_chain()
                    labels = ["epsH1", "gam1", "nu1", "const","epsH2", "gam2", "nu2"]
                    for i in range(ndim):
                        ax[i].plot(samples[:, :, i], "k", alpha=0.3)
                        ax[i].set_xlim(0, len(samples))
                        ax[i].set_ylabel(labels[i])
                        ax[i].yaxis.set_label_coords(-0.1, 0.5)
                    ax[-1].set_xlabel("step number")
                    plt.show()

                #plotting corner plot
                if plotting:
                    fig,ax = plt.subplots()
                    flat_samples = sampler.get_chain(discard=100, flat=True)
                    corner.corner(flat_samples,
                                  labels=["epsH1", "gam1", "nu1", "const","epsH2", "gam2", "nu2"],
                                  truths=params)
                    plt.show()

                #Estimating error from one sigma
                e_params = []
                for j in range(ndim):
                    percentiles = np.percentile(flat_samples[:,j], [16,50,84])
                    q = np.diff(percentiles)
                    e_params.append(max(q))
                e_params = np.array(e_params)


                #Finding amplitude of peak
                lim_a = point[0] - reg
                lim_b = point[0] + reg


                params1 = params[:4]
                
                params2 = []
                for j in params[4:]:
                    params2.append(j)
                params2.append(params[3])
                params2 = np.array(params2)

                
                integral1,e_integral1 = int_of_peak(lim_a,lim_b, params1,e_params1)
                integral2,e_integral2 = int_of_peak(lim_a,lim_b, params2,e_params2)


                
                #Saving info
                amplitudes[k,i] = integral1
                e_amplitudes[k,i] = e_integral1
                amplitudes[k,weak_idx] = integral2
                e_amplitudes[k,weak_idx] = e_integral2


                for j,e_param in enumerate(e_params[:4]):
                    e_ml_params[k,i,j] = e_param

                for j,e_param in enumerate(e_params[4:]):
                    e_ml_params[k,weak_idx,j] = e_param
                    
                ml_params[2,weak_idx,3] = e_params[3]#The constant for the weak peak

                

                
            else:
                nwalkers,ndim = 15,4
                iterations = 1000
                burnin = 100
                params = np.array([epsH1,gam1,nu1,const1])
                pos = params + 1e-4*np.random.randn(nwalkers,ndim)

                
                if True: #if using multiprocessing
                    with Pool(6) as pool:
                        sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability,
                                                        args=(f[idx_peak],p[idx_peak]),pool=pool)
                        sampler.run_mcmc(pos,iterations,progress=True)
                else:
                    sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                                        log_probability,
                                                    args=(f[idx_peak],p[idx_peak]))
                    sampler.run_mcmc(pos,iterations,progress=True)
                    
                    
                samples = sampler.get_chain()
                labels = ["epsH", "gam", "nu", "const"]
                flat_samples = sampler.get_chain(discard=burnin, flat=True)

                #plotting sampling
                if plotting:
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
                if plotting:
                    fig,ax = plt.subplots()
                    corner.corner(flat_samples, labels=['epsH', 'gam', 'nu', 'const'],
                                  truths=params)
                    plt.show()



                #Estimating error from one sigma
                e_params = []
                for j in range(ndim):
                    percentiles = np.percentile(flat_samples[:,j], [16,50,84])
                    q = np.diff(percentiles)
                    e_params.append(max(q))
                e_params = np.array(e_params)


                #Finding amplitude of peak
                lim_a = point[0] - reg
                lim_b = point[0] + reg
                
                integral,e_integral = int_of_peak(lim_a,lim_b, params,e_params)

                #Saving
                amplitudes[k,i] = integral
                e_amplitudes[k,i] = e_integral

                e_ml_params[k,i,:] = e_params


    if saving_data:
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        np.save(path_to_save + 'individual_peaks_e_max_like',e_ml_params)
        np.save(path_to_save + 'individual_peaks_amplitude',amplitudes)
        np.save(path_to_save + 'individual_peaks_e_amplitude',e_amplitudes)

            


if True:
    analyse_power('KIC10454113',saving_data = True, plotting = True) 

if False:  
    analyse_power('KIC9025370',saving_data = True, plotting = True)

if False: 
    analyse_power('KIC12317678',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = False)

if False: 
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











