import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seismology_functions as sf
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from peak_bagging_tool import simple_peak_bagging
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

    
def rand_Gauss(shape):
    out_array = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = np.random.normal(0,0.01)
            while (-0.01 > val) or (0.01 < val):
                val = np.random.normal(0,1)
            out_array[i,j] = val
    return out_array
        
    

##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,
                  reg=40,peak_type='all'):
    
    '''
    Function for finding the error of the parameters of individual modes, as well as
    the amplitude of the peaks, i.e. the area under the curve.

    parameters:
        - saving_data   : bool, save the data or not
        - plotting      : bool, plot the data or not
        - reg           : float, the area before and after the peaks to include amplitude est.
        - peak_type     : str, [all, mode1, mode02,mode0,mode2].
                            Describes what peaks you want to fit
    returns:
        - Nothing is returned. If saving_data = True, then errorsare saved to file
            as 'individual_peaks_e_max_like'
    '''
    
    ##################### Importing info: ###########################
    ID_idx = np.where(ID == IDs)[0]
    if len(ID_idx) != 1:
        print('ID was given wrong')
        print('The ID should be among the following:')
        print(IDs)
        return 0

    


    #Power spec filtered
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    power_spec_df = pd.read_csv(path_to_save + 'filt_power_spec.txt').to_numpy()
    f,p = power_spec_df[:,0], power_spec_df[:,1]

    #importing peak info
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    guess_points = np.load( path_to_save + f'individual_peaks_eye_peaktype_{peak_type}.npy')

    ml_params = np.load(path_to_save + f'individual_peaks_max_like_peaktype_{peak_type}.npy')
    MCMC_ml_params = np.zeros(shape=(ml_params.shape[0],ml_params.shape[1],ml_params.shape[2]))
    e_ml_params = np.zeros(shape=(ml_params.shape[0],ml_params.shape[1],ml_params.shape[2]))
    amplitudes = np.zeros(shape=(ml_params.shape[0],ml_params.shape[1]))
    e_amplitudes = np.zeros(shape=(ml_params.shape[0],ml_params.shape[1]))
    
        
    ################## Errors through MCMC fitting ################################
    #Running through every order:
    for i in range(len(guess_points[0])):


        nr_peaks = 0
        mode0 = guess_points[1][i]
        mode1 = guess_points[0][i]
        mode2 = guess_points[2][i]

        epsH1, nu1, gam, const = ml_params[0,i,:]
        epsH0,  nu0, gam0, const0 = ml_params[1,i,:]
        epsH2, nu2, gam2, const2 = ml_params[2,i,:]

        print('Parameters from fit: ', ml_params[0,i,:],ml_params[1,i,:],ml_params[2,i,:])
        #checking if fit was good
        points = []
        if nu1 != 0: 
            points.append(mode1)
            if nu0 != 0:
                points.append(mode0)
                if nu2 != 0:
                    points.append(mode2)
        else:
            continue
        nr_peaks = int(len(points)-1) #int, nr of peaks in order
        
        #isolating the part of the spectrum we want to fit
        if nr_peaks == 0:
            idx_peak = (( mode1 - reg < f ) &
                     ( mode1 + reg > f ) )
            params = np.array([epsH1,nu1,gam,const])
            labels = ['epsH','nu','gam','const']
        if nr_peaks == 1:
            idx_peak = (( mode0 - reg < f ) &
                     ( mode1 + reg > f ) )
            params = np.array([epsH1,nu1,epsH0,nu0,gam,const])
            labels = ['epsH1','nu1','epsH0','nu0','gam','const']
            
        if nr_peaks == 2:
            idx_peak = (( mode2 - reg < f ) &
                     ( mode1 + reg > f ) )
            params = np.array([epsH1,nu1,epsH0,nu0, epsH2,nu2,gam,const])
            labels = ['epsH1','nu1','epsH0','nu0', 'epsH2','nu2','gam','const']

        nwalkers,ndim = 18,len(params)
        iterations = 1000
        burnin = 200

        bounds = []
        for j in range(nr_peaks+1):
            j *= 2
            bounds.append((0,20))#epshH
            bounds.append((params[j+1]-100,params[j+1]+100))#nu
        bounds.append((0,20))# gam
        bounds.append((0,10))#const

        #Drawing from normal ball truncated to +/- 0.1
        #pos = params + params*rand_Gauss((nwalkers,ndim))
        pos = params + 1e-4 * np.random.randn(nwalkers,ndim)
        with Pool(7) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                            sf.log_probability_N,
                                            args=(f[idx_peak],p[idx_peak],bounds),
                                            pool=pool)
            sampler.run_mcmc(pos,iterations,progress=True)


        #plotting sampling
        if plotting:
            fig, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            samples = sampler.get_chain()
            
            for j in range(ndim):
                ax[j].plot(samples[:, :, j], "k", alpha=0.3)
                ax[j].set_xlim(0, len(samples))
                ax[j].set_ylabel(labels[j])
                ax[j].vlines(burnin,min(samples[:, :, j].flatten()),
                             max(samples[:, :, j].flatten()), ls='--', color='red')
                ax[j].yaxis.set_label_coords(-0.1, 0.5)
            ax[-1].set_xlabel("step number")
            plt.show()

        #plotting corner plot
        flat_samples = sampler.get_chain(discard=burnin, flat=True)
        corner.corner(flat_samples,
                          labels=labels,
                          truths=params,
                          quantiles=[0.16, 0.5, 0.84])
        if plotting: plt.show()
        plt.close()

        #Estimating error from one sigma
        MCMC_params = []
        e_params = []
        for j in range(ndim):
            percentiles = np.percentile(flat_samples[:,j], [16,50,84])
            q = np.diff(percentiles)
            e_params.append(max(q))
            MCMC_params.append(percentiles[1])
        MCMC_params = np.array(MCMC_params)
        e_params = np.array(e_params)

        
        e_params1 = [e_params[0],e_params[1],e_params[-2],e_params[-1]]
        params1 = [params[0],params[1],params[-2],params[-1]]
        MCMC_params1 = [MCMC_params[0],MCMC_params[1],
                        MCMC_params[-2],MCMC_params[-1]]
        e_ml_params[0,i,:] = e_params1
        MCMC_ml_params[0,i,:] = MCMC_params1
        if nr_peaks >0:
            e_params0 = [e_params[2],e_params[3],e_params[-2],e_params[-1]]
            params0 = [params[2],params[3],params[-2],params[-1]]
            MCMC_params0 = [MCMC_params[2],MCMC_params[3],
                            MCMC_params[-2],MCMC_params[-1]]
            
            e_ml_params[1,i,:] = e_params0
            MCMC_ml_params[1,i,:] = MCMC_params0
        if nr_peaks >1:
            e_params2 = [e_params[4],e_params[5],e_params[-2],e_params[-1]]
            params2 = [params[4],params[5],params[-2],params[-1]]
            MCMC_params2 = [MCMC_params[4],MCMC_params[5],
                            MCMC_params[-2],MCMC_params[-1]]
            e_ml_params[2,i,:] = e_params2
            MCMC_ml_params[2,i,:] = MCMC_params2
            

        #Finding amplitude of peak
        
        lim_a = mode1 - reg
        lim_b = mode1 + reg
        integral1,e_integral1 = sf.int_of_peak(lim_a,lim_b, params1,e_params1)
        
        #Saving info
        amplitudes[0,i] = integral1
        e_amplitudes[0,i] = e_integral1
            
        if nr_peaks > 0:
            
            lim_a = mode0 - reg
            lim_b = mode0 + reg
            integral0,e_integral0 = sf.int_of_peak(lim_a,lim_b, params0,e_params0)

            #Saving info
            amplitudes[1,i] = integral0
            e_amplitudes[1,i] = e_integral0
            
        if nr_peaks >1:

            lim_a = mode2 - reg
            lim_b = mode2 + reg
            integral2,e_integral2 = sf.int_of_peak(lim_a,lim_b, params2,e_params2)
            
            #Saving info
            amplitudes[2,i] = integral2
            e_amplitudes[2,i] = e_integral2


    
    if saving_data:
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        np.save(path_to_save + f'individual_peaks_e_max_like_peaktype_{peak_type}',e_ml_params)
        np.save(path_to_save + f'individual_peaks_MCMC_peaktype_{peak_type}',MCMC_ml_params)
        np.save(path_to_save + f'individual_peaks_amplitude_peaktype_{peak_type}',amplitudes)
        np.save(path_to_save + f'individual_peaks_e_amplitude_peaktype_{peak_type}',e_amplitudes)

            


if False: #all, mode1, mode02
    analyse_power('KIC10454113',saving_data = True,reg=40,
                  plotting = False,peak_type='mode02')

if False: #all, mode1, mode02
    analyse_power('KIC9693187',saving_data = True,reg=30,
                  plotting = False,peak_type='all') 

if True:  
    analyse_power('KIC9025370',saving_data = True, reg=20,
                  plotting = True,peak_type='all')

if False: #all, mode1,mode02
    analyse_power('KIC12317678',saving_data = True,reg=20,
                  plotting = True,peak_type='mode02')

if False: 
    analyse_power('KIC4914923',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True)

'''
[EPIC236224056,EPIC246696804,EPIC249570007,EPIC230748783,EPIC212617037]
'''











