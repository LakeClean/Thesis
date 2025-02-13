import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seismology_functions as sf
from peak_bagging_tool import simple_peak_bagging
from scipy.ndimage import gaussian_filter
import lmfit
from scipy.optimize import minimize


'''
Script for determining the positions of peak in the power spectrum.
This script should be run after "find_ACF.py" has been run.
'''

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



##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,
                  find_ind_peaks = True, ask = True,reg=20,peak_type='all'):
    '''
    Function for finding parameters of individual modes

    parameters:
        - saving_data   : bool, save the data or not
        - plotting      : bool, plot the data or not
        - find_ind_peaks: bool, if true call simple_peak_bagging function
        - ask           : bool, ask for confirmation of the fit of the individual order
        - reg           : float, the area before and after the peaks to include in fit
        - peak_type     : str, [all, mode1, mode02,mode0,mode2].
                            Describes what peaks you want to fit
                            When all: modes should be selected as mode0:x, mode1:z mode2:v
                            When mode1: modes should be selected as mode1:z
                            When mode0: modes should be selected as mode0:z
                            When mode2: modes should be selected as mode2:z
                            When mode02: modes should be selected as mode0:z, mode2,x
    returns:
        - Nothing is returned. If saving_data = True, then fitting parameters are saved to file
            as 'individual_peaks_max_like'
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
    power_spec_df = pd.read_csv(path_to_save + 'filt_power_spec.txt').to_numpy()
    f,p = power_spec_df[:,0], power_spec_df[:,1]

    print('########################################################')
    print(f'Analysing: {ID}')
    print('########################################################')
    print(f'You are currently saving as peak_type = {peak_type}')
    print('When all: modes should be selected as l=0:x, l=1:z l=2:v')
    print('When mode1: modes should be selected as l=1:z')
    print('When mode02: modes should be selected as l=0:z, l=2,x')
    print('########################################################')
    print('Note that mode l=1 should be to the right of mode 0 and 2')
    print('########################################################')
    
    
    #################### Analysing: ################################

    if find_ind_peaks:
        guess_points = simple_peak_bagging(f,p)
        if saving_data:
            path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
            np.save(path_to_save + f'individual_peaks_eye_peaktype_{peak_type}',guess_points)

    else:
        try:
            path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
            guess_points = np.load( path_to_save + f'individual_peaks_eye_peaktype_{peak_type}.npy')

        except:
            print('You have not found the individual peaks by eye yet!')
            return 0

    #maximum likelihood
    ml_params = np.zeros(shape=(guess_points.shape[0],guess_points.shape[1],4))

    #logfile for fitting:
    fitlogfile = open(f'fitting_logs/fitting_log_{ID}_{peak_type}.txt','w')
    fitlogfile.write('The fits of the following orders failed\n')
    

    #################### Parameters through maximum likelihood: #########################

    #Running through every order:
    for i in range(len(guess_points[0])):
    
        
        mode0 = guess_points[1][i]
        mode1 = guess_points[0][i]
        mode2 = guess_points[2][i]
        
        #checking if point exists
        points = []
        if mode1 != 0: 
            points.append(mode1)
            if mode0 != 0:
                points.append(mode0)
                if mode2 != 0:
                    points.append(mode2)
        else:
            continue
        
        nr_peaks = int(len(points)-1) #int, nr of peaks in order


    
        #isolating the part of the spectrum we want to fit
        if nr_peaks == 0:
            idx_peak = (( mode1 - reg < f ) &
                     ( mode1 + reg > f ) )
        if nr_peaks == 1:
            idx_peak = (( mode0 - reg < f ) &
                     ( mode1 + reg > f ) )
            
        if nr_peaks == 2:
            idx_peak = (( mode2 - reg < f ) &
                     ( mode1 + reg > f ) )

        #plotting frequency window
        if plotting:
            fig, ax = plt.subplots()
            ax.plot(f[idx_peak],p[idx_peak])

        #Fitting with least squares:
        print(f'{nr_peaks} peaks close!')
        
        params = lmfit.Parameters()
        for j in range(nr_peaks+1):
            params.add(f'epsH{j}',value=5,min=0,max=10)
            params.add(f'nu{j}',points[j])

        params.add(f'gam',value=3,min=0,max=10)
        params.add('const',value=0)

        fit = lmfit.minimize(sf.mode_N_res, params, args=(f,p),
                     xtol=1.e-8,ftol=1.e-8,max_nfev=500)
        
        print(lmfit.fit_report(fit,show_correl=False))

        values = []
        for val in list(fit.params.values()):
            values.append(val.value)

        if plotting :
            ax.plot(f[idx_peak],
                sf.mode_N(f[idx_peak],values),
                label='least squares fit', color='red')


        #We estimate parameters with maximum likelihood
        nll = lambda *args: -sf.log_likelihood_N(*args)
        initial = abs(np.array(values))
        bounds = []
        for j in range(nr_peaks+1):
            j *= 2
            bounds.append((0,10))#epshH
            bounds.append((initial[j+1]-reg,initial[j+1]+reg))#nu
        bounds.append((0,10))# gam
        bounds.append((0,10))#const

        soln = minimize(nll, initial, args=(f[idx_peak],p[idx_peak]),
                        bounds=bounds,
                        method = 'Nelder-Mead')
        
        for init,bound in zip(initial,bounds):
            if (init > bound[1]) or (init < bound[0] ):
                print(initial)
                fitlogfile.write(f'order: {i}\n')

        print('Parameters of max likelihood fit: ', soln.x)
        if plotting:
            ax.plot(f[idx_peak], sf.mode_N(f[idx_peak],soln.x),
                    label='maximized likelihood')
            plt.show()


        
        
        if ask:
            if 'n' == input('Is this a good fit?(y/n)'):
                print('Fit was rejected')
                fitlogfile.write(f'order: {i}\n')
                continue
            else:
                print('Fit was accepted')
        

        ml_params[0,i,:] = np.array([soln.x[0],soln.x[1],soln.x[-2],soln.x[-1]])
        if nr_peaks > 0:
            ml_params[1,i,:] = np.array([soln.x[2],soln.x[3],soln.x[-2],soln.x[-1]])
            if nr_peaks >1:
                ml_params[2,i,:] = np.array([soln.x[4],soln.x[5],soln.x[-2],soln.x[-1]])


    fitlogfile.close()
    if saving_data:
        path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
        np.save(path_to_save + f'individual_peaks_max_like_peaktype_{peak_type}',ml_params)


if False:#all, mode1, mode02
    analyse_power('KIC9693187',saving_data = True, plotting = True,
                  find_ind_peaks = False,ask = True,peak_type='mode02',reg=30)           

if False:#all, mode1, mode02
    analyse_power('KIC10454113',saving_data = True, plotting = True,
                  find_ind_peaks = False,ask = True,peak_type='all',reg=40) 

if False:#all, mode1, mode02
    analyse_power('KIC9025370',saving_data = True, plotting = True,
                  find_ind_peaks = False,ask = True,peak_type='all',reg=20)

if True: #all, mode1, mode02
    analyse_power('KIC12317678',saving_data = True, plotting = True,
                  find_ind_peaks = False,ask = True,peak_type='mode02',reg=20)

if False: #all, mode1, mode02
    analyse_power('KIC4914923',saving_data = True, plotting = True,
                  find_ind_peaks = True,ask = True,peak_type='mode02',reg=20)

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











