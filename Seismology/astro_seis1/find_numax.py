import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seismology_functions as sf
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
import lmfit
from ophobningslov import *
from scipy.optimize import minimize



master_path = '/usr/users/au662080'

'''
Script for finding numax and alternative values of dnu.
This script should be run after find_error_in_peaks.py
'''



#importing log file
log_file_path = f'{master_path}/Speciale/data/Seismology/analysis/'
log_file_path += 'log_file.txt'
log_df = pd.read_csv(log_file_path)

IDs = log_df['ID'].to_numpy()
#data_types = log_df['data_type'].to_numpy()
#numax1_guesss = log_df['numax1_guess'].to_numpy()
#numax2_guesss = log_df['numax2_guess'].to_numpy()
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

    out_df = pd.DataFrame(out_dict)
    
    out_df.to_csv(path_to_save,index = False)


def Gaussian(x,theta):
    A,mu,std,floor = theta
    return A*np.exp(-0.5*((x-mu)/std)**2) + floor

def Gaussian_res(params,x,y,weight=1):
    '''
    Returns the residual between Gaussian and data weighted by the error.
    '''
    values = []
    for i in list(params.values()):
        values.append(i.value)
        
    res = y - Gaussian(x,values)
    return res * weight
'''
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    floor = params['floor'].value
    res = y - Gaussian(x,a,b,c,floor)
    return res * weight
'''

def my_diff(xs,e_xs):
    '''
    Function for finding the mean difference of an iterable xs given the errrors of xs.
    returns difference and error in difference as:
    diff, e_diff

    '''
    diffs = []
    e_diffs = []
    for i in range(len(xs)-1):
        diffs.append(xs[i+1] - xs[i])
        e_diffs.append(np.sqrt(e_xs[i+1]**2 + e_xs[i]**2))
    n_e_diffs = e_diffs/sum(e_diffs) # The normalized error
    diffs = np.array(diffs)
    e_diffs = np.array(e_diffs)
    diff = sum(diffs*n_e_diffs)
    e_diff = np.std(diffs) * np.sqrt(sum(n_e_diffs**2))
    return diff, e_diff

##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,):
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

    print('Analysing ',ID)

    #################### Analysing: ################################
    
    #Finding numax:
    alt_dnus_out = np.zeros(3) #an alternative dnu estimate
    e_alt_dnus_out = np.zeros(3) # error in alternative dnu estimate
    numax_est = np.zeros(3) #positions of gaussian for modes
    e_numax_est = np.zeros(3) #'error' of gaussian for modes

    
    fig, ax = plt.subplots(1,3)
    modes = ['all','mode02','mode1']
    marker_mode = {'all':['s','^','*'], 'mode02':['^','*','.'],
                   'mode1':['s','.','.']} #marker for plotting
    color_mode = {'all':['blue','orange','red'],
                  'mode02':['orange','red','pink'],
                  'mode1':['blue','pink','pink']}#color for plotting
    label_mode = {'all':['1','0','2'], 'mode02':['0','2','1'],
                   'mode1':['1','2','2']} #label for plotting
    for j, mode in enumerate(modes):
        
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        ml_params = np.load(path_to_save + f'individual_peaks_max_like_peaktype_{mode}.npy')
        e_ml_params = np.load(path_to_save + f'individual_peaks_e_max_like_peaktype_{mode}.npy')
        guess_points = np.load(path_to_save + f'individual_peaks_eye_peaktype_{mode}.npy')
        amplitudes = np.load(path_to_save + f'individual_peaks_amplitude_peaktype_{mode}.npy')
        e_amplitudes = np.load(path_to_save + f'individual_peaks_e_amplitude_peaktype_{mode}.npy')
        
        for k in range(len(ml_params)):
            



            #sorting in orders of n based on the guesses by eye:
            sorted_amplitudes = [x for y,x in sorted(zip(guess_points[k],amplitudes[k]))]
            sorted_e_amplitudes = [x for y,x in sorted(zip(guess_points[k],e_amplitudes[k]))]
            sorted_nus = [x for y,x in sorted(zip(guess_points[k],ml_params[k,:,1]))]
            sorted_e_nus = [x for y,x in sorted(zip(guess_points[k],e_ml_params[k,:,1]))]
            sorted_gams = [x for y,x in sorted(zip(guess_points[k],ml_params[k,:,2]))]
            sorted_e_gams = [x for y,x in sorted(zip(guess_points[k],e_ml_params[k,:,2]))]
            orders = np.arange(0,len(sorted_amplitudes),1)

            #Removing zeros
            idx = np.array(sorted_nus) > 0
            sorted_amplitudes = np.array(sorted_amplitudes)[idx]
            sorted_e_amplitudes = np.array(sorted_e_amplitudes)[idx]
            sorted_nus = np.array(sorted_nus)[idx]
            sorted_e_nus = np.array(sorted_e_nus)[idx]
            sorted_gams = np.array(sorted_gams)[idx]
            sorted_e_gams = np.array(sorted_e_gams)[idx]
            orders = orders[idx]

            if len(sorted_amplitudes)<2:
                continue

            #Sorting into points that are adjacent (cluster)
            clusters = [[]]
            e_clusters = [[]]
            for i in range(len(orders)-1):
                if orders[i+1] - orders[i] == 1:
                    clusters[-1].append(sorted_nus[i])
                    e_clusters[-1].append(sorted_e_nus[i])
                if orders[i+1] - orders[i] != 1:
                    clusters.append([sorted_nus[i]])
                    e_clusters.append([sorted_e_nus[i]])



            #Averaging each of the clusters
            alt_dnus = []
            e_alt_dnus = []
            for cluster, e_cluster in zip(clusters,e_clusters):
                if len(cluster)>1:
                    alt_dnu_i, e_alt_dnu_i = my_diff(cluster,e_cluster)
                    alt_dnus.append(alt_dnu_i)
                    e_alt_dnus.append(e_alt_dnu_i)


            alt_dnus = np.array(alt_dnus)
            e_alt_dnus = np.array(e_alt_dnus)

            

            
            if len(alt_dnus)>1 :
                n_e_alt_dnus = e_alt_dnus / sum(e_alt_dnus) #normalized errors
                alt_dnu = sum(n_e_alt_dnus * alt_dnus)
                e_alt_dnu = np.std(alt_dnus) * np.sqrt(sum(n_e_alt_dnus**2))
            else:
                alt_dnu = alt_dnus[0]
                e_alt_dnu = e_alt_dnus[0]
                
            alt_dnus_out[k] = alt_dnu
            e_alt_dnus_out[k] = e_alt_dnu
            
        

            
            if len(sorted_amplitudes) <4:
                continue

            
            #Fitting Gaussian
            params = lmfit.Parameters()
            params.add('A',value=max(sorted_amplitudes))
            params.add('mu', value=np.mean(sorted_nus))
            params.add('std', value=np.std(sorted_nus))
            params.add('floor',value=0)

            fit = lmfit.minimize(Gaussian_res, params,
                                 args=(sorted_nus,sorted_amplitudes,sorted_e_amplitudes),
                                 xtol=1.e-8,ftol=1.e-8,max_nfev=500)
            
            #print(lmfit.fit_report(fit,show_correl=False))
            A = fit.params['A'].value
            mu = fit.params['mu'].value
            std = fit.params['std'].value
            floor = fit.params['floor'].value

            print('covariance:',fit.covar)
            try:
                e_A,e_mu,e_std,e_floor = np.sqrt(np.diagonal(fit.covar))
                #e_mu,e_std,e_floor = np.sqrt(np.diagonal(fit.covar))
            except:
                e_A,e_mu,e_std,e_floor = 0,0,0,0
                #e_mu,e_std,e_floor = 0,0,0

            if mode == 'mode1':
                numax_est[k] = mu
                e_numax_est[k] = e_mu


            #plotting
            freq_space = np.linspace(min(sorted_nus),max(sorted_nus),100)
            #norm = max(Gaussian(freq_space,[A,mu,std,floor]))
            
            ax[j].plot(freq_space,Gaussian(freq_space,[A,mu,std,floor]),ls='--', alpha=0.4,
                                      zorder=3,color=color_mode[mode][k],
                       label=f'Gaussian fit. Peak = {np.round(mu,1)}+/-{np.round(e_mu,1)}')

            ax[j].errorbar(sorted_nus,sorted_amplitudes,sorted_e_amplitudes,capsize=2,ls='',
                           label=f'l modes: {label_mode[mode][k]}, alt_dnu = {np.round(alt_dnu,1)}+/-{np.round(e_alt_dnu,1)}',
                          color=color_mode[mode][k],marker=marker_mode[mode][k])
            
            ax[j].set_title(f'Fit method: {mode}')
            ax[j].set_xlabel('Frequency of modes')
            ax[j].set_ylabel('Amplitude [area of modes]')
        
        

    for i in range(3): ax[i].legend(loc='best',bbox_to_anchor=(0.5,0.0))
    plt.show()

    print('numax: ', numax_est, e_numax_est)
    print('alternative dnus: ', alt_dnus_out,e_alt_dnus_out)
    

    if saving_data: save_data(ID, 'numax',
                                  [numax_est,e_numax_est],
                                  ['numax','numax_error'])
    
    if saving_data: save_data(ID, 'alt_dnu', [alt_dnus_out,e_alt_dnus_out],
                                  ['alt_dnu', 'e_alt_dnu'])





if True:
    analyse_power('KIC10454113',saving_data = True, plotting = True)

if True: 
    analyse_power('KIC9693187',saving_data = True, plotting = True) 

if True:  
    analyse_power('KIC9025370',saving_data = True, plotting = True)

if True: 
    analyse_power('KIC12317678',saving_data = True, plotting = True)

if True: 
    analyse_power('KIC4914923',saving_data = True, plotting = True)











