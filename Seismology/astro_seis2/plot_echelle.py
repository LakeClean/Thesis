import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import glob
import seismology_functions as sf
from matplotlib.widgets import Button, Slider
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import lmfit
from scipy.integrate import quad
from ophobningslov import *
from scipy.optimize import minimize


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

    #print(out_dict)
    out_df = pd.DataFrame(out_dict)
    #print(out_df)
    out_df.to_csv(path_to_save,index = False)


##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,
                  peak_type='all'):
    '''
    Function for finding plotting echelle. Should be run after
    '''
    
    
    ID_idx = np.where(ID == IDs)[0]
    if len(ID_idx) != 1:
        print('ID was given wrong')
        print('The ID should be among the following:')
        print(IDs)
        return 0

    print('Analysing ',ID,)

    #Power spec filtered
    path_to_save = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
    power_spec_df = pd.read_csv(path_to_save + 'filt_power_spec.txt').to_numpy()
    f,p = power_spec_df[:,0], power_spec_df[:,1]

    #importing numax and dnu info:
    numax, e_numax = pd.read_csv(path_to_save + 'numax.txt').to_numpy()[0]
    dnu = pd.read_csv(path_to_save + 'dnu.txt').to_numpy()[0][0]

    #Importing parameters of fit to individual modes 
    ml_params = np.load(path_to_save + f'individual_peaks_max_like_peaktype_{peak_type}.npy')


    # Calculating echelle diagram:
    FWHM = numax/2
    start = numax-FWHM
    stop = numax+FWHM
    A,B,EchelleVals=sf.Echellediagram(p,f,dnu,start,
                                      stop,no_repeat=1)
    VMIN = np.min(EchelleVals)
    VMAX = np.max(EchelleVals)
    

    #plot echelle
    if plotting:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)

        ax.plot([0,dnu],[numax,numax],ls='--',
                alpha=0.4,color='blue',zorder=1,
                label=f'numax: {np.round(numax,1)}+/-{np.round(e_numax,1)}')
            
        image = ax.imshow(EchelleVals,aspect='auto',
                  extent=[0,dnu,start,stop],
                  norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
                  cmap=cm.gray_r,zorder=0)

       
        colors = ['green', 'red', 'yellow']
        scatter_tags = []
        
        for k in range(len(ml_params)):
            peak_idx = ml_params[k,:,1] > 0
            scatter_tag = ax.scatter(ml_params[k,:,1][peak_idx]%dnu,
                                     ml_params[k,:,1][peak_idx],
                                     color=colors[k])
            scatter_tags.append(scatter_tag)


                                    
        ax.set_xlabel(f' Frequency mod dnu={np.round(dnu,2)}muHz')
        ax.set_ylabel('Frequency muHz')
        ax.set_title(ID)
        ax.legend()
        

        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(
            ax=axfreq,
            label='dnu#',
            valmin=dnu-1,
            valmax=dnu+1,
            valinit=dnu,
            valstep=0.01
        )
        def update(val):
            A,B,new_EchelleVals=sf.Echellediagram(p,f,val,start,
                                      stop,no_repeat=1)
            image.set_data(new_EchelleVals)

            for k in range(len(scatter_tags)):
                peak_idx = ml_params[k,:,1] > 0
                scatter_tags[k].set_offsets(np.column_stack((ml_params[k,:,1][peak_idx]%val,
                                                             ml_params[k,:,1][peak_idx])))
                scatter_tags[k].set_facecolor(colors[k])
            
            fig.canvas.draw_idle()
            
        slider.on_changed(update)


        plt.show()

    

    #### Collapsed power: ####
    idx_limit = (f<2*numax)
    f, p = f[idx_limit], p[idx_limit]
    df = f[10] - f[9]
    Norders = 5
    ncollap = int(dnu/df)
    collapse = np.zeros(ncollap)
    start = np.argmin(np.abs(f-(numax-dnu*Norders/2)))
    for j in range(Norders):
        collapse += p[start+ncollap*j:start+ncollap*(j+1)]
    collapsed_normed = collapse/Norders
    eps_axis = sf.crange2(1,2,
                         f[start:start+ncollap]/dnu \
                         - int(f[start]/dnu) +1)
    idx_sort = np.argsort(eps_axis)
    eps_axis = (eps_axis[idx_sort]-1)*dnu
    collapsed_normed = collapsed_normed[idx_sort]


    smoothed_collapsed = gaussian_filter(collapsed_normed, sigma=5)
    
    if plotting:
        fig, ax = plt.subplots()
        ax.plot(eps_axis,collapsed_normed, label='normed collapsed power')
        ax.plot(eps_axis,smoothed_collapsed, label='Smoothed')
        ax.set_ylabel('collapsed power')
        ax.set_xlabel('Frequency')
        ax.legend()
        plt.show()






if False:
    analyse_power('KIC10454113', plotting = True)

if False:
    analyse_power('KIC9693187', plotting = True) 

if False:  
    analyse_power('KIC9025370', plotting = True)

if False: 
    analyse_power('KIC12317678', plotting = True)

if False: 
    analyse_power('KIC4914923', plotting = True)

'''
[EPIC236224056,EPIC246696804,EPIC249570007,EPIC230748783,EPIC212617037]
'''











