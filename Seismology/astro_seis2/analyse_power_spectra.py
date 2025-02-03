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
    print('Analysing ',ID, '|  numax1 guess:', numax1_guess)
    if numax1_guess == numax2_guess: numax_guess = numax1_guess


    if data_type == 'dat': #data is timeseries format
        
        data_df = pd.read_csv(data_path,skiprows=13,delimiter=r"\s+").to_numpy()
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
    df = f[10]-f[9]
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

    if saving_data: save_data(ID, 'ACF', [acf2,lagvec2],
                                  ['ACF', 'lagvec2'])
    if saving_data: save_data(ID, 'weighted_ACF', [acf2_w,lagvec2],
                                  ['ACF_w', 'lagvec2'])

    #Finding maximums in ACF:
    dnu_guess = sf.dnu_guess(numax_guess)
    dnu_peak1 = sf.find_maximum(dnu_guess,lagvec1,acf1/acf1[1],
                                dnu_guess*0.9,dnu_guess*1.1)[0]
    
    dnu_peak2 = sf.find_maximum(dnu_guess,lagvec2,acf2/acf2[1],
                                dnu_guess*0.9,dnu_guess*1.1)[0]
    print(f'dnu from unweighted PSD: {dnu_peak1}')
    print(f'dnu from weighted PSD: {dnu_peak2}')
    if saving_data: save_data(ID, 'dnu', [[dnu_peak1],[dnu_peak2]],
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

    # Calculating echelle diagram:
    FWHM = numax_guess/2
    start = numax_guess-FWHM
    stop = numax_guess+FWHM
    A,B,EchelleVals=sf.Echellediagram(p_filt,f,dnu_peak1,start,
                                      stop,no_repeat=1)
    VMIN = np.min(EchelleVals)
    VMAX = np.max(EchelleVals)
    

    #plot echelle
    if plotting:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        colors = ['green', 'red', 'yellow']

        for i,n_est in enumerate(numax_est):
            ax.plot([0,dnu_peak1],
                    [n_est,n_est],
                    ls='--',alpha=0.4,color=colors[i])
            
        image = ax.imshow(EchelleVals,aspect='auto',
                  extent=[0,dnu_peak1,start,stop],
                  norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
                  cmap=cm.gray_r,zorder=0)

       

        scatter_tags = []
        for k in range(len(ml_params)):
            scatter_tag = ax.scatter(ml_params[k,:,2]%dnu_peak1,
                                        ml_params[k,:,2], color=colors[k])
            scatter_tags.append(scatter_tag)


                                    
        ax.set_xlabel(f' Frequency mod dnu={np.round(dnu_peak1,2)}muHz')
        ax.set_ylabel('Frequency muHz')
        ax.set_title(ID)
        

        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(
            ax=axfreq,
            label='dnu#',
            valmin=dnu_peak1-1,
            valmax=dnu_peak1+1,
            valinit=dnu_peak1,
            valstep=0.01
        )
        def update(val):
            A,B,new_EchelleVals=sf.Echellediagram(p_filt,f,val,start,
                                      stop,no_repeat=1)
            image.set_data(new_EchelleVals)

            for k in range(len(scatter_tags)):
                scatter_tags[i].set_offsets(np.column_stack((ml_params[k,:,2]%val,
                                                             ml_params[k,:,2])))
                scatter_tags[i].set_facecolor(colors[i])
            
            fig.canvas.draw_idle()
            
        slider.on_changed(update)


        plt.show()

    

    #### Collapsed power: ####
    idx_limit = (f<2*numax_guess)
    f, p_filt = f[idx_limit], p[idx_limit]
    Norders = 5
    ncollap = int(dnu_peak1/df)
    collapse = np.zeros(ncollap)
    start = np.argmin(np.abs(f-(numax_guess-dnu_peak1*Norders/2)))
    for j in range(Norders):
        collapse += p_filt[start+ncollap*j:start+ncollap*(j+1)]
    collapsed_normed = collapse/Norders
    eps_axis = sf.crange2(1,2,
                         f[start:start+ncollap]/dnu_peak1 \
                         - int(f[start]/dnu_peak1) +1)
    idx_sort = np.argsort(eps_axis)
    eps_axis = (eps_axis[idx_sort]-1)*dnu_peak1
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

    


    #### ridge_centroids: ####
    '''
    f = data[:,0] #muHz
    p = data[:,1] #ppm**2 / muHz
    p_bg = sf.logmed_filter(f,p,filter_width=0.2)
    p = p/p_bg
    
    smo_width = 5
    df = f[10]-f[9]
    win = int(smo_width/df)
    if win%2==0:
        win+=1
    p_filt = sf.filtering(p,win)
    
    start_fre = numax*0.6
    stop_fre = numax*1.4
    dnus = np.linspace(dnu_peak1-2*dnu_peak1/100,dnu_peak1+2*dnu_peak1/100, 7)
    oaps_fre, av_oaps, oaps = sf.OAPS(f,p_filt,start_fre,
                                      stop_fre,dnus,K=4)
    fig,ax = plt.subplots(2,1)
    for k in range(oaps.shape[0]):
        ax[0].plot(oaps_fre,oaps[k,:])
        
    ax[1].plot( oaps_fre,av_oaps)
    
    av_oaps_filt = filtering(av_oaps,win)
    FWHM = numax/2
    start = numax-FWHM
    stop = numax+FWHM
    A,B,Echellediagram(p_filt,f,dnu_peak1,start,stop,no_repreat=1)
    fig,ax = plt.subplots()
    ax.imshow(EchelleVals,aspect='auto',
                  extent=[0,dnu_peak1,start,np.max(B)+0.5*dnu_peak1],
                  norm=LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
                  cmap=cm.gray_r,zorder=0)

    for i in range(len(B)):
        try:
            idx_cut = (oaps_fre>B[i]-dnu_peak1/2) & (oaps_fre<B[i]+dnu_peak1/2)
            oap_fcut, oap_valcut = oaps_fre[idx_cut], av_oaps_filt[idx_cut]
            eps_vals = np.mod(oap_fcut,dnu_peak1)/dnu_peak1 + 1
            idx_cut_l0 = (eps_vals<1.5)
            f0 = oap_fcut[idx_cut_l0][np.argmax(oap_valcut[idx_cut_l0])]
            idx_cut_l1 = (eps_vals>1.5)
            f1 = oap_fcut[idx_cut_l1][np.argmax(oap_valcut[idx_cut_l1])]
            idx_cut_l2 = (eps_vals<1.5) & (oap_fcut<f0-0.8)
            f2 = oap_fcut[idx_cut_l2][np.argmax(oap_valcut[idx_cut_l2])]
            ax.scatter(np.mod(f1,dnu_peak1),f1,color='C1')
            ax.scatter(np.mod(f2,dnu_peak1),f2,color='C2')
            ax.scatter(np.mod(f0,dnu_peak1),f0,color='C0')
        except (IndexError, ValueError):
            continue

            
    plt.show()
    '''




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











