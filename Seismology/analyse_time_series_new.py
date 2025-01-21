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
from matplotlib.widgets import Button, Slider
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
from peak_bagging_tool import simple_peak_bagging
from scipy.ndimage import gaussian_filter
import lmfit



master_path = '/usr/users/au662080'

'''
Data is given in either filtered Power spectra or as filtered timeseries.
We first deal with the timeseries:

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

    print(out_dict)
    out_df = pd.DataFrame(out_dict)
    print(out_df)
    out_df.to_csv(path_to_save,index = False)


def Gaussian(x,a,b,c,floor):
    return a*np.exp(-0.5*((x-b)/c)**2) + floor

def Gaussian_res(params,x,y):
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    floor = params['floor'].value
    res = y - Gaussian(x,a,b,c,floor)
    return res


##############################################################################

def analyse_power(ID,saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = True):
    '''
    Function for finding numax and deltanu of given target.
    Also does a lot of plotting.
    Give data as timeseries. and allow only for target with double visble seismic signal.
    '''
    
    
    ID_idx = np.where(ID == IDs)[0]
    if len(ID_idx) != 1:
        print('ID was given wrong')
        print('The ID should be among the following:')
        print(IDs)
        return 0

    data_type = data_types[ID_idx][0]
    numax1_guess = numax1_guesss[ID_idx][0]
    numax2_guess = numax2_guesss[ID_idx][0]
    data_path = data_paths[ID_idx][0]
    print('Analysing ',ID, '|  numax1 guess:', numax1_guess, '|  numax2 guess:', numax2_guess)


    if data_type == 'dat': #data is timeseries format
        
        data = pd.read_csv(data_path,skiprows=13,delimiter=r"\s+").to_numpy()
        rel_time, flux, e_flux = data[:,0], data[:,1], data[:,2]
        idx = (0<e_flux)
        rel_time, flux, e_flux = rel_time[idx], flux[idx], e_flux[idx]
        PDS = powerspectrum(rel_time,flux,flux_err=e_flux,weighted=True)
        f, p0 = PDS.powerspectrum(scale='powerdensity')
        
        if plotting:
            fig,ax = plt.subplots()
            ax.plot(rel_time,flux)
            ax.set_title(ID)
            ax.set_xlabel('Truncated barycentric JD')
            ax.set_ylabel('Relative flux')
            plt.show()

    if data_type == 'pow': #data is power spectrum format
        data_df = pd.read_csv(data_path,skiprows=15,delimiter=r"\s+").to_numpy()
        f = data_df[:,0] #muHz
        p0 = data_df[:,1] #ppm**2 / muHz
        if saving_data: save_data(ID, 'power_spec', [f,p0],
                                  ['Frequency', 'power'])


    #### filtering: ####
    df = f[10]-f[9] #Assuming that window is constant in frequency
    win = int(1/df)
    if win%2==0: win+=1                              
    if filtering:
        #p_bg = sf.logmed_filter(f,p0,filter_width=0.2)
        #p = p0/p_bg
        p = p0
        p_filt = sf.filtering(p,win)
        if saving_data: save_data(ID, 'filt_power_spec', [f,p_filt],
                                  ['Frequency', 'filt_power'])


    #### Plotting filtered and unfiltered psd: 
    if plotting:
        fig, ax = plt.subplots()
        ax.plot(f,p0, label='raw')
        ax.plot(f,p_filt, label='Epan. filtered')
        
        ax.set_xlabel(r'frequency [$\mu$Hz]')
        ax.set_ylabel(r'power [ppm^2 / $\mu$Hz]')
        ax.set_title(f'Power spec of {ID}')
        ax.legend()
        plt.show()


    mus_12 = [] #numax_estimates
    stds_12 = [] #numax_error 
    lxnus_12 = [] #positions of individual modes
    dnu_peaks_12 = [] #delta nus
    for numax_guess in [numax1_guess,numax2_guess]:
        #### Finding ACF: ####
        env_scale=1
        sigma_env = env_scale * numax_guess/(4*np.sqrt(2*np.log(2)))

        HWHM = numax_guess/4
        idx1 = (f<numax_guess + 1.5*HWHM) & (f> numax_guess -1.5*HWHM)
        idx2 = (f<numax_guess + 3*HWHM) & (f> numax_guess -3*HWHM)

        weight = 1 / (sigma_env * np.sqrt(2*np.pi)) * np.exp(-(f - numax_guess)**2 / (2*sigma_env**2) )
        pds_w = p*weight

        acf1 = sf.autocorr_fft(p_filt[idx1])
        acf2 = sf.autocorr(p_filt[idx1])

        acf1_w = sf.autocorr_fft(pds_w[idx1])
        acf2_w = sf.autocorr(pds_w[idx1])

        df = f[10] - f[9]
        lagvec1 = np.arange(len(acf1))*df
        lagvec2 = np.arange(len(acf2))*df

        if saving_data: save_data(ID, f'ACF_nmguess{numax_guess}', [acf2,lagvec2],
                                      ['ACF', 'lagvec2'])
        if saving_data: save_data(ID, 'weighted_ACF_nmguess{numax_guess}', [acf2_w,lagvec2],
                                      ['ACF_w', 'lagvec2'])

        #Finding maximums in ACF:
        dnu_guess = sf.dnu_guess(numax_guess)
        dnu_peak1 = sf.find_maximum(dnu_guess,lagvec1,acf1/acf1[1],
                                    dnu_guess*0.9,dnu_guess*1.1)[0]
        
        dnu_peak2 = sf.find_maximum(dnu_guess,lagvec2,acf2/acf2[1],
                                    dnu_guess*0.9,dnu_guess*1.1)[0]
        print(f'dnu from unweighted PSD: {dnu_peak1}')
        print(f'dnu from weighted PSD: {dnu_peak2}')
        if saving_data: save_data(ID, 'dnu_nmguess{numax_guess}', [[dnu_peak1],[dnu_peak2]],
                                      ['dnu_peak1', 'dnu_peak2'])
        
                                  


        smoothed = gaussian_filter(p[idx1], sigma=6)
        #plotting
        if plotting:
            fig,ax = plt.subplots(2,2)
            ax[0,0].set_title(ID)

            
            ax[0,0].plot(f[idx2], p[idx2])
            ax[0,0].plot(f[idx1], p[idx1])
            ax[0,0].plot(f[idx1],smoothed)
            ax[0,0].set_xlabel(r'frequency [$\mu$Hz]')
            ax[0,0].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


            ax[0,1].plot(f[idx2], pds_w[idx2])
            ax[0,1].plot(f[idx1], pds_w[idx1])
            ax[0,1].plot(f[idx2],np.max(pds_w[idx1])*weight[idx2]/np.max(weight),
                         color='k',ls='--',label='weight')
            ax[0,1].set_xlabel(r'frequency [$\mu$Hz]')
            ax[0,1].set_ylabel(r'power [$ppm^2 / \mu$Hz]')
            ax[0,1].legend()


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
            for i in range(7):
                i+=1
                ax[1,1].vlines(x=dnu_peak2*i,ymin=-1,ymax=1,ls='--',color='k')
                ax[1,1].vlines(x=dnu_peak2*i,ymin=-1,ymax=1,ls='--',color='k')

            fig.tight_layout()
            plt.show()


        if find_ind_peaks:
            guess_points, gauss_params, mode_params = simple_peak_bagging(f[idx2],
                                                                    p[idx2])

            for k in range(3):
                if saving_data: save_data(ID, f'individual_peaks_mode{k+1}_nmguess{numax_guess}',
                                      [[x[0] for x in mode_params[k]],
                                       [x[1] for x in mode_params[k]],
                                       [x[2] for x in mode_params[k]],
                                       [x[3] for x in mode_params[k]],
                                       [x[4] for x in mode_params[k]]],
                                      ['eps','H','gam','nu','const'])
                
                if saving_data: save_data(ID, f'individual_peaks_gauss{k+1}_nmguess{numax_guess}',
                                      [[x[0] for x in gauss_params[k]],
                                       [x[1] for x in gauss_params[k]],
                                       [x[2] for x in gauss_params[k]]],
                                      ['std','mu','floor'])


                if saving_data: save_data(ID, f'individual_peaks_eye{k+1}_nmguess{numax_guess}',
                                      [[x[0] for x in guess_points[k]],
                                       [x[1] for x in guess_points[k]]],
                                      ['x', 'y'])


        
        else:
            mode_params = [[],[],[]]
            gauss_params = [[],[],[]]
            guess_points = [[],[],[]]
            ind_peaks_path = f'{master_path}/Speciale/data/Seismology/analysis/{ID}/'
            
            for k in range(3):
                mode_path = ind_peaks_path+f'individual_peaks_mode{k+1}_nmguess{numax_guess}.txt'
                ind_peaks_mode_df = pd.read_csv(mode_path).to_numpy()
                mode_params[k].append(ind_peaks_mode_df)

                gauss_path = ind_peaks_path+f'individual_peaks_gauss{k+1}_nmguess{numax_guess}.txt'
                ind_peaks_gauss_df = pd.read_csv(gauss_path).to_numpy()
                gauss_params[k].append(ind_peaks_gauss_df)

                eye_path = ind_peaks_path+f'individual_peaks_eye{k+1}_nmguess{numax_guess}.txt'
                ind_peaks_eye_df = pd.read_csv(eye_path).to_numpy()
                guess_points[k].append(ind_peaks_eye_df)
                
 


            
        
        '''
        #Finding numax:

        mus = []
        stds = []
        lxnus = []
        fig, ax = plt.subplots()
        for k in range(len(mode_params)):
            
            if find_ind_peaks:
                if len(mode_params[k]) == 0:
                    break
                epss = np.array(mode_params[k])[:,0]
                Hs = np.array(mode_params[k])[:,1]
                gams = np.array(mode_params[k])[:,2]
                nus = np.array(mode_params[k])[:,3]
                consts = np.array(mode_params[k])[:,4]
            else:
                if len(mode_params[k][0]) == 0:
                    break
                epss = mode_params[k][0][:,0]
                Hs = mode_params[k][0][:,1]
                gams = mode_params[k][0][:,2]
                nus = mode_params[k][0][:,3]
                consts = mode_params[k][0][:,4]
                
            lxnus.append(nus)

            amplitude = [x for y, x in sorted(zip(nus,Hs*epss))]
            frequency = [y for y, x in sorted(zip(nus,Hs*epss))]


            #Fitting Gaussian
            params = lmfit.Parameters()
            params.add('a',value=max(amplitude))
            params.add('b', value=numax_guess)
            params.add('c', value=np.std(frequency))
            params.add('floor',value=0)

            fit = lmfit.minimize(Gaussian_res, params,
                                 args=(frequency,amplitude),
                                 xtol=1.e-8,ftol=1.e-8,max_nfev=500)
            print(lmfit.fit_report(fit,show_correl=False))
            a = fit.params['a'].value
            b = fit.params['b'].value
            c = fit.params['c'].value
            floor = fit.params['floor'].value
            stds.append(c)
            mus.append(b)

            

            freq_space = np.linspace(min(frequency),max(frequency),100)
            norm = max(Gaussian(freq_space,a,b,c,floor))
            
            ax.plot(freq_space,Gaussian(freq_space,a,b,c,floor)/norm,ls='--',
                                      zorder=3)

            ax.scatter(frequency,amplitude/norm)
            ax.set_xlabel('Center freuqency')
            ax.set_ylabel('Height * mode visibility (normed)')
        plt.show()
        

        if saving_data: save_data(ID, 'numax_nmguess{numax_guess}',
                                      [mus,stds],
                                      ['numax','numax_error'])

        mus_12.append(mus) #numax_estimates
        stds_12.append(stds) #numax_error 
        lxnus_12.append(lxnus) #positions of individual modes
        dnu_peaks_12.append(dnu_peak1) #delta nus
    

    # Calculating echelle diagram:
    #FWHM = numax_guess/2
    start = numax1_guess-numax1_guess/2
    stop = numax2_guess+numax2_guess/2
    A,B,EchelleVals=sf.Echellediagram(p_filt,f,dnu_peaks_12[0],start,
                                      stop,no_repeat=1)
    VMIN = np.min(EchelleVals)
    VMAX = np.max(EchelleVals)
    

    #plot echelle
    if plotting:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        colors = ['green', 'red']

        image = ax.imshow(EchelleVals,aspect='auto',
                  extent=[0,dnu_peak1,start,stop],
                  norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
                  cmap=cm.gray_r,zorder=0)

       

        scatter_tags = []
        for k in range(len(lxnus_12)):
            tags = []
            for i in range(len(lxnus)):
                scatter_tag = ax.scatter(lxnus_12[k][i]%dnu_peaks_12[k],
                                        lxnus_12[k][i], color=colors[k])
                tags.append(scatter_tag)
            scatter_tags.append(tags)


                                    
        ax.set_xlabel(f' Frequency mod dnu={np.round(dnu_peak1,2)}muHz')
        ax.set_ylabel('Frequency muHz')
        ax.set_title(ID)
        

        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(
            ax=axfreq,
            label='dnu#',
            valmin=min(dnu_peaks_12)-1,
            valmax=max(dnu_peaks_12)+1,
            valinit=dnu_peaks_12[0],
            valstep=0.01
        )
        def update(val):
            A,B,new_EchelleVals=sf.Echellediagram(p_filt,f,val,start,
                                      stop,no_repeat=1)
            image.set_data(new_EchelleVals)

            for k in range(len(scatter_tags)):
                for i in range(len(scatter_tags[k])):
                    scatter_tags[k][i].set_offsets(np.column_stack((lxnus_12[k][i]%val,
                                                                    lxnus_12[k][i])))
                    scatter_tags[k][i].set_facecolor(colors[k])
            
            fig.canvas.draw_idle()
            
        slider.on_changed(update)


        plt.show()
    

    

    #### Collapsed power: ####
    for i, numax_guess in enumerate([numax1_guess, numax2_guess]):
        idx_limit = (f<2*numax_guess)
        f, p_filt = f[idx_limit], p[idx_limit]
        Norders = 5
        ncollap = int(dnu_peaks_12[i]/df)
        collapse = np.zeros(ncollap)
        start = np.argmin(np.abs(f-(numax_guess-dnu_peaks_12[i]*Norders/2)))
        for j in range(Norders):
            collapse += p_filt[start+ncollap*j:start+ncollap*(j+1)]
        collapsed_normed = collapse/Norders
        eps_axis = sf.crange2(1,2,
                             f[start:start+ncollap]/dnu_peaks_12[i] \
                             - int(f[start]/dnu_peaks_12[i]) +1)
        idx_sort = np.argsort(eps_axis)
        eps_axis = (eps_axis[idx_sort]-1)*dnu_peaks_12[i]
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
    '''



	  					
        
if True:
    analyse_power('KIC4260884',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = False)
if True:
    analyse_power('KIC4457331',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = False)
if True:
    analyse_power('KIC9652971',saving_data = True, plotting = True,
                  filtering = True,find_ind_peaks = False)









