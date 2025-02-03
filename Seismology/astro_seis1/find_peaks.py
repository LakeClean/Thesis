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


'''
Script for determining the positions and amplitudes and their associated errors for the peaks
in the power spectrum.
This script should be run after "find_ACF.py" has been run.

'''





master_path = '/usr/users/au662080'





#importing log file
log_file_path = f'{master_path}/Speciale/data/Seismology/analysis/'
log_file_path += 'log_file.txt'
log_df = pd.read_csv(log_file_path)

IDs = log_df['ID'].to_numpy()
#data_types = log_df['data_type'].to_numpy()
#data_paths = log_df['data_path'].to_numpy()



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

    #data_type = data_types[ID_idx][0]
    #data_path = data_paths[ID_idx][0]


    #Power spec unfiltered
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    power_spec_df = pd.read_csv(path_to_save + 'power_spec.txt').to_numpy()
    f,p = power_spec_df[:,0], power_spec_df[:,1]
    

    
    #################### Analysing: ################################

    
    if find_ind_peaks:
        guess_points, gauss_params, mode_params, region = simple_peak_bagging(f,p)
        if saving_data:
            path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
            np.save(path_to_save + 'individual_peaks_mode',mode_params)
            np.save(path_to_save + 'region',region)
            np.save(path_to_save + 'individual_peaks_eye',guess_points)

    else:
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        guess_points = np.load( path_to_save + 'individual_peaks_eye.npy')
        mode_params = np.load(path_to_save + 'individual_peaks_mode.npy')
        #region = np.load(path_to_save + 'region.npy')
        

    #maximum likelihood
    ml_params = np.zeros(shape=(mode_params.shape[0],mode_params.shape[1],4))

    #logfile for fitting:
    fitlogfile = open(f'fitting_log_{ID}.txt','w')
    fitlogfile.write('The fits of the following peaks failed\n')
    
    reg = 20
    #################### Parameters through maximum likelihood: #########################
    for k in range(len(guess_points)-1):
        for i, point in enumerate(guess_points[k]):
            

            close = False #bool, whether there is a peak very close
            x_point = point[0]

            #checking if point exists
            if x_point == 0: 
                continue

            
            if k == 1:
                for j in range(len(mode_params[2][0])):
                    v_point = guess_points[2][j][0]
                    eps2_guess, H2_guess, gam2_guess, nu2_guess, const2_guess = mode_params[2,j,:]
                    
                    if abs(x_point - v_point) < 20:
                        close = True
                        weak_idx = j
                        break

            
            #Simple guess from peakbagging tool
            eps_guess,H_guess,gam_guess,nu_guess,const_guess = mode_params[k,i,:]
            #initial = np.array([eps_guess*H_guess,gam_guess,nu_guess,const_guess])

            #limiting power spectrum to around peak:
            idx_peak = (point[0] - reg<f) & (point[0] + reg>f)

            #plotting frequency window
            if plotting:
                fig, ax = plt.subplots()
                ax.plot(f[idx_peak],p[idx_peak])

            #Fitting with least squares:
            if close:
                print('close!')
                params = lmfit.Parameters()
                params.add('epsH1',value=eps_guess*H_guess)
                params.add('gam1',value=gam_guess)
                params.add('nu1',value = x_point)
                params.add('const',value=const_guess)
                params.add('epsH2',value=eps2_guess*H2_guess)
                params.add('gam2',value=gam2_guess)
                params.add('nu2',v_point)

                fit = lmfit.minimize(mode_double_res, params, args=(f[idx_peak],p[idx_peak]),
                                     xtol=1.e-8,ftol=1.e-8,max_nfev=500)
                
                print(lmfit.fit_report(fit,show_correl=False))

                epsH1 = fit.params['epsH1'].value
                gam1 = fit.params['gam1'].value
                nu1 = fit.params['nu1'].value
                const = fit.params['const'].value
                epsH2 = fit.params['epsH2'].value
                gam2 = fit.params['gam2'].value
                nu2 = fit.params['nu2'].value

                if plotting :
                    ax.plot(f[idx_peak],
                        mode_double(f[idx_peak],[epsH1,gam1,nu1,const,epsH2,gam2,nu2]),
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
                if plotting:
                    ax.plot(f[idx_peak],mode(f[idx_peak],[epsH,gam,nu,const]),
                        label='least squares fit', color='red')

        

            #We estimate parameters with maximum likelihood

            if close:
                nll = lambda *args: -log_likelihood_double(*args)
                initial = np.array([epsH1, gam1, nu1,const,epsH2, gam2, nu2])

                soln = minimize(nll, initial, args=(f[idx_peak],p[idx_peak]),
                                bounds=[(0.000001,10),(-100,100),
                                        (0.000001,10000),(0.000001,20),
                                        (0.000001,10),
                                        (-100,100),(0.000001,10000)],
                                method = 'Nelder-Mead')
                for init,bound in zip(initial,[(0.000001,10),(-100,100),
                                    (0.000001,10000),(0.000001,20)]):
                    if (init > bound[1]) or (init < bound[0] ):
                        print(initial)
                        fitlogfile.write(f'{point[0]}\n')
                if plotting:
                    ax.plot(f[idx_peak], mode_double(f[idx_peak],soln.x),
                            label='maximized likelihood')


            else:

                nll = lambda *args: -log_likelihood(*args)
                initial = np.array([epsH, gam, nu,const])

                soln = minimize(nll, initial, args=(f[idx_peak],p[idx_peak]),
                                bounds=[(0.000001,10),(-100,100),
                                        (0.000001,10000),(0.000001,20)],
                                method = 'Nelder-Mead')
                
                for init,bound in zip(initial,[(0.000001,10),(-100,100),
                                        (0.000001,10000),(0.000001,20)]):
                    if (init > bound[1]) or (init < bound[0] ):
                        print(initial)
                        fitlogfile.write(f'{point[0]}\n')
                if plotting:
                    ax.plot(f[idx_peak], mode(f[idx_peak],soln.x),label='maximized likelihood')
            if plotting:
                smoothed = gaussian_filter(p[idx_peak], sigma=6)
                ax.plot(f[idx_peak],smoothed,label='smoothed')
                plt.show()

            


            #Saving info
            if close:
                for j,param in enumerate(soln.x[:4]):
                    ml_params[k,i,j] = param
                    
                for j,param in enumerate(soln.x[4:]):
                    ml_params[2,weak_idx,j] = param

                ml_params[2,weak_idx,3] = soln.x[3]

            else:
                for j, param in enumerate(soln.x):
                    ml_params[k,i,j] = param

    fitlogfile.close()
    if saving_data:
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        np.save(path_to_save + 'individual_peaks_max_like',ml_params)


            


if True:
    analyse_power('KIC10454113',saving_data = True, plotting = True,find_ind_peaks = False) 

if False:  
    analyse_power('KIC9025370',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True)

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











