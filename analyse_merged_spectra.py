import shazam
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import glob
#import pickle #For saving figures as matplotlib figures
from astropy.modeling import models, fitting


#import template
template_dir = '/home/lakeclean/Documents/speciale/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
tfl_RG = template_data['arcturus']
tfl_MS = template_data['solarflux']
twl = template_data['wavelength']


#fig, ax  = plt.subplots()
#ax.plot(twl,tfl_MS)
#plt.show()

#import the RGB info
lines = open('/home/lakeclean/Documents/speciale/NOT/Target_names_and_info.txt').read().split('\n')[1:14]
all_target_names = [] # Just a list of the names
RGBs = [] # names of known RGB stars
for line in lines:
    line = line.split()
    all_target_names.append(line[0])

    if len(line)>5: #noting the RG stars
        if line[5]=='(RGB)':
            RGBs.append(line[0])


def analyse_spectrum(file, start_wl=0, end_wl=100000,
            template='MS', bin_size=200,
                     
                     normalize_bl = np.array([]),normalize_poly=2,normalize_gauss=True,
                     normalize_lower=0.5,normalize_upper=1.5,
                     
                     crm_iters = 1, crm_q = [99.0,99.9,99.99],
                     
                     resample_dv=1.0, resample_edge=0.0,
                     
                     getCCF_rvr=401, getCCF_ccf_mode='full',
                     
                     getBF_rvr=401, getBF_dv=1.0,
                     
                     rotbf2_fit_fitsize=30,rotbf2_fit_res=60000,rotbf2_fit_smooth=2.0,
                     rotbf2_fit_vsini1=5.0,rotbf2_fit_vsini2=5.0,rotbf2_fit_vrad1=-30.0,
                     rotbf2_fit_vrad2=-17.0,rotbf2_fit_ampl1=0.5,rotbf2_fit_ampl2=0.5,
                     rotbf2_fit_print_report=True,rotbf2_fit_smoothing=True,
                     
                     rotbf_fit_fitsize=30,rotbf_fit_res=60000,rotbf_fit_smooth=2.0,
                     rotbf_fit_vsini=5.0,rotbf_fit_print_report=True,
                     
                     use_SVD=False,SB_type=1,
                     show_plots=True, save_plots=True, save_data = True,
                     show_bin_plots=False,save_bin_info=False): 
    '''
    Analyses a raw merged spectra. Merged as in each order has been merged.
    The function needs the path to the merged fits file.
    It then stores the analysis of the spectra in existing directory.
    
    :params:
        file      :str, path of the fits file
        start_wl  :int, starting wavelength [Ångstrom]
        end_wl    :int, ending wavelength [Ångstrom]
        template  :str, either 'MS' or 'RG'
        bin_size  :int, size of wavelength bin
        SB_type   :int, either 1 or 2 works atm. when use_SVD=True

        rvr       : integer, range for RVs in km/s
        fitsize   : float, fitsize in km/s - only fit `fitsize` part of the rotational profile
        res       : float, resolution of spectrograph - affects width of peak
        smooth    : float, smoothing factor - sigma in Gaussian smoothing
        
    
    :returns:
        rvs       :np.array, radial velocities for each
                        'order'/subdivision [km/s]
                        
        vsini     :np.array, rotaitonal velocity for each
                        'order'/subdivision [km/s]
    Notice that many functions are called from the module shazam. Parameters for these functions
    can set globally for the whole analyse_spectrm function in the function call.
    They are name (function from shazam)_parameter

    Method:
        - Spectrum is read in
        - Name is checked to see of RGB
        - Template is read in
        - Raw spectrum is plotted
        - for loop for every bin as specfified by start_wl, end_wl and bin_size
            - Select the right bin
            - normalize spectrum with shazamm.normalize
            - pick out 95% percentile and further normalize with this
            - remove cosmic rays with shazam.crm
            - fit line to normalized spectrum to see if flat
            - resample and flip spectrum and template with shazam.resample
            -
    '''

    data = pyfits.getdata(file)
    header = pyfits.getheader(file)

    epoch_name = header['TCSTGT'].strip(' ') #name of target
    epoch_date = header['DATE_OBS'].strip(' ')   #date of fits creation

    
    if epoch_name in RGBs:
        tfl = tfl_RG
    else:
        tfl = tfl_MS

    path = '/home/lakeclean/Documents/speciale/target_analysis/' + epoch_name +'/' + epoch_date

    #######################################################################
    def save_datas(datas,labels,title):
        f = open(path + f"/data/{title}.txt",'w')
        result1 = ''
        for i in labels:
            result1 += f'{i},'
            
        f.write(f'{result1[:-1]}\n')
        
        for i in range(len(datas[0])):
            result2 = ''
            for j in range(len(datas)):
                result2 += f'{datas[j][i]},'
            f.write(f'{result2[:-1]}\n')
            
        f.close()
    ########################################################################

    #We have to start from the starting wavelength
    lam = header['CDELT1']*np.arange(header['NAXIS1'])+header['CRVAL1']
    if start_wl<lam[0]:
        start_wl = lam[0]
    if end_wl > lam[-1]:
        end_wl = lam[-1]

    #Test of normalizing spectrum before binning:

    lam, data = shazam.normalize(lam,data,normalize_bl,normalize_poly,
                                    normalize_gauss,normalize_lower,normalize_upper)
    

    
        
    #Raw spectrum is plotted
    fig, ax = plt.subplots()
    x_label, y_label, title ='wavelength [Å]', 'flux (raw)', 'Merged spectrum'
    xs, ys = lam, data
    ax.plot(xs,ys)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
    ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
    if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                               dpi='figure', format='svg')
    if show_plots: plt.show()
    if save_data: save_datas([xs,ys],[x_label,y_label],plot_title.replace(' ','_'))
    plt.close()

    
    
    n_bins = int((end_wl-start_wl)//bin_size)

    epoch_nwls = [] #binned norm wl 
    epoch_nfls = [] #binned norm fl

    epoch_rf_wls = [] #binned resamp and flipped wl 
    epoch_rf_fls = [] #binned resamp and flipped fl
    epoch_rf_tfls = [] #binned resamp and flipped template

    epoch_rvs = [] #binned rvs for BF or ccf
    epoch_bf = [] #binned BF
    epoch_smoothed_bf = [] #binned smoothed BF

    epoch_ccf = [] #binned BF

    epoch_rv = np.zeros(n_bins)
    epoch_vsini = np.zeros(n_bins)

    bin_wls = np.arange(n_bins)* bin_size +start_wl
    slopes = np.zeros(n_bins)

    for i in np.arange(n_bins):
        begin = start_wl + bin_size*i
        end = begin + bin_size

        if end > end_wl: # break for loop if we reach specified wl
            break

        #Pick out correct wl range
        index = np.where((lam>begin) & ( lam<end))[0]
        wl = lam[index]
        print(len(wl))
        fl = data[index]
        if save_data: save_datas([wl,fl],['wavelength [Å]', 'flux (raw)'],
                                 f"bin_{i}_size_{bin_size}_raw_spectrum")
        
        if np.mean(fl)<0.001: print('flux is very low') 

        #normalize
        nwl, nfl = shazam.normalize(wl,fl, normalize_bl,normalize_poly,
                                    normalize_gauss,normalize_lower,normalize_upper)

        nfl = nfl / np.median(nfl[np.where(np.percentile(nfl,95)<nfl)[0]])#The flux of the normalized flux that is above 95% percentile
        
        if save_data: save_datas([nwl,nfl],['wavelength [Å]', 'flux (norm)'],
                                 f"bin_{i}_size_{bin_size}_normalized")
        epoch_nwls.append(nwl)
        epoch_nfls.append(nfl)

        #Remove cosmic rays
        nwl, nfl = shazam.crm(nwl, nfl, crm_iters, crm_q)

        #check whether norm is flat:
        fit = fitting.LinearLSQFitter()
        line_init = models.Linear1D()
        fitted_line = fit(line_init, nwl, nfl)
        slope = fitted_line.slope.value
        slopes[i] = slope
        
        
        
        #Resample and flip:
        
        r_wl, rf_fl, rf_tl = shazam.resample(nwl,nfl,twl,tfl, resample_dv, resample_edge)
        if save_data: save_datas([r_wl, rf_fl, rf_tl],
                                 ['wavelength(resampled) [Å]',
                                  'flux (resampled and flipped)',
                                  'template flux (resampled and flipped)']
                                  ,f"bin_{i}_size_{bin_size}_resampled_flipped")
        epoch_rf_wls.append(r_wl)
        epoch_rf_fls.append(rf_fl)
        epoch_rf_tfls.append(rf_tl)
        

        #If we want to get BF from SVD
        if use_SVD:
            
            rvs,bf = shazam.getBF(rf_fl,rf_tl,getBF_rvr, getBF_dv)
            epoch_rvs.append(rvs)
            epoch_bf.append(bf)

            if SB_type == 1:
                fit, model, bfgs = shazam.rotbf_fit(rvs,bf,rotbf_fit_fitsize,rotbf_fit_res,
                                                    rotbf_fit_smooth,rotbf_fit_vsini,
                                                    rotbf_fit_print_report)
                rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
            elif SB_type == 2:
                fit, model, bfgs = shazam.rotbf2_fit(rvs,bf, rotbf2_fit_fitsize,rotbf2_fit_res,
                                                     rotbf2_fit_smooth, rotbf2_fit_vsini1,
                                                     rotbf2_fit_vsini2,rotbf2_fit_vrad1,
                                                     rotbf2_fit_vrad2,rotbf2_fit_ampl1,
                                                     rotbf2_fit_ampl2,rotbf2_fit_print_report,
                                                     rotbf2_fit_smoothing)
                
                rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value

            else:
                print('The SB type was given wrong: Should be int 1 or 2')
                
            
            epoch_smoothed_bf.append(bfgs)
            epoch_rv[i] = rv + header['VHELIO'] 
            epoch_vsini[i] = vsini
            
            if save_data: save_datas([rvs,bf,bfgs,model],
                                     ['wavelength [km/s]', 'broadening function',
                                      'smoothed_bf', 'Fitted model'],
                                     f"bin_{i}_size_{bin_size}_broadening_function")
            
        #If we want approximate BF from cross correlation
        else:
            #cross correlate
            rvs, ccf = shazam.getCCF(rf_fl,rf_tl,getCCF_rvr, getCCF_ccf_mode)
            epoch_rvs.append(rvs)
            epoch_ccf.append(ccf)
            #rvs_test, fit = shazam.getfit_CCF(rvs,ccf) #Showing the fit to the CCF

            if SB_type == 1:
                fit, model, bfgs = shazam.rotbf_fit(rvs,ccf,rotbf_fit_fitsize,rotbf_fit_res,
                                                    rotbf_fit_smooth,rotbf_fit_vsini,
                                                    rotbf_fit_print_report)
                rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
                
            elif SB_type == 2:
                fit, model, bfgs = shazam.rotbf2_fit(rvs,ccf,rotbf2_fit_fitsize,rotbf2_fit_res,
                                                     rotbf2_fit_smooth, rotbf2_fit_vsini1,
                                                     rotbf2_fit_vsini2,rotbf2_fit_vrad1,
                                                     rotbf2_fit_vrad2,rotbf2_fit_ampl1,
                                                     rotbf2_fit_ampl2,rotbf2_fit_print_report,
                                                     rotbf2_fit_smoothing)
                rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value

            else:
                print('The SB type was given wrong: Should be int 1 or 2')

            if save_data: save_datas([rvs,ccf,model],
                                     ['wavelength [km/s]', 'cross correlation','fit to ccf'],
                                     f"bin_{i}_size_{bin_size}_cross_correlation") # model should be changed maybe

            #get rv:
            rv, vsini = shazam.getRV(rvs,ccf)
            
            epoch_rv[i] = rv + header['VHELIO']
            epoch_vsini[i] = vsini



        # Plotting for every bin:
        if show_bin_plots:
            fig, ax = plt.subplots(3,1,figsize=(10,8))
            #raw spectrum
            ax[0].plot(wl,fl,label='raw flux')
            ax[0].set_xlabel('Wavelength [Å]')
            ax[0].set_ylabel('flux (raw)')
            
            #normalized spectrum
            ax[1].plot(nwl,nfl,label='normalized')
            ax[1].set_xlabel('Wavelength [Å]')
            ax[1].set_ylabel('normalized flux')
            ax[1].plot(nwl,fitted_line(nwl),label=f'linear fit: {slope}')
            ax[1].legend()

            #resampled and fliped template and spectrum
            ax[2].plot(r_wl,rf_fl,label='spectrum')
            ax[2].plot(r_wl,rf_tl,label='template')
            ax[2].set_xlabel('Resampled wavelength [Å]')
            ax[2].set_ylabel('Flipped flux')
            ax[2].legend()

            fig.suptitle(f'Bin #{i} ' + epoch_name + ' ' + epoch_date)
            
            
            fig, ax = plt.subplots(figsize=(8,3))
            fig.suptitle(f'Bin #{i} ' + epoch_name + ' ' + epoch_date)
            #broadening/cross correlation functions:
            if use_SVD:
                ax.plot(rvs,bf,label='Broadening Function')
                ax.plot(rvs,bfgs,label='Smoothed broadening Function')
                ax.plot(rvs,model,label='Fit to BF')
                ax.set_xlabel('radial velocity [km/s]')
                ax.set_ylabel('Broadening Function')
                ax.legend()
            else:
                ax.plot(rvs,ccf,label='Cross correlation')
                ax.plot(rvs,bfgs,label='Smoothed')
                ax.set_xlabel('radial velocity [km/s]')
                ax.set_ylabel('Cross correlation')
                rvs_test, fit = shazam.getfit_CCF(rvs,ccf,SB_type=SB_type) #Showing the fit to the CCF
                ax.plot(rvs,fit,label='gaussian fit')
                ax.plot(rvs,model,label='lmfit')
                ax.legend()

            
            plt.show()
            plt.close()

    # Plotting bfs or ccfs together:
    if use_SVD:
        fig, ax = plt.subplots()
        x_label, y_label, title ='rvs [km/s]', 'broadening function', 'broadening function'
        k=0
        for xs, ys,smoothed in zip(epoch_rvs,epoch_bf,epoch_smoothed_bf):
            ax.plot(xs,ys+k)
            ax.plot(xs,smoothed+k)
            k+=0.1
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0,n_bins*0.1)
        ax2 = ax.twinx()  
        ax2.set_ylim(bin_wls[0], bin_wls[-1])
        plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
        ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
        if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                                   dpi='figure', format='svg')
        if show_plots: plt.show()
        plt.close()

    else:
        fig, ax = plt.subplots()
        x_label, y_label, title ='rvs [km/s]', 'cross correlation', 'cross correlation'
        k = 0
        for xs, ys in zip(epoch_rvs,epoch_ccf):
            ax.plot(xs,ys+k)
            k+=1
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0,n_bins)
        ax2 = ax.twinx()  
        ax2.set_ylim(bin_wls[0], bin_wls[-1])
        plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
        ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
        if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                                   dpi='figure', format='svg')
        if show_plots: plt.show()
        plt.close()

    
    # Plotting radial velocities measured:
    fig, ax = plt.subplots()
    x_label, y_label='bin_wl', 'Radial Velocity [km/s]'
    title = 'radial velocity per bin'
    xs, ys = bin_wls,epoch_rv
    ax.scatter(xs,ys,label='rv')
    ax.plot([xs[0],xs[-1]],[np.mean(ys),np.mean(ys)],label='mean',color='r')
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
    ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
    if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                               dpi='figure', format='svg')
    if show_plots: plt.show()
    if save_data: save_datas([xs,ys],[x_label,y_label],plot_title.replace(' ','_'))
    plt.close()
    
    ################ Testing bins sizes: ##############
    #Std:
    raw_std = np.std(epoch_rv)
    index_no_out = np.where(abs(np.mean(epoch_rv) - epoch_rv) < np.std(epoch_rv))[0]
    no_out_std = np.std(epoch_rv[index_no_out])
    index_limited = np.where((bin_wls> 3750) & (bin_wls<6500))[0]
    limited_std = np.std(epoch_rv[index_limited])
    
    if save_data: save_datas([bin_wls,slopes],
                             ['bin_wl','slope of normalization'],
                             f'slope_per_bin_size{bin_size}')

    
    if save_bin_info:
        std_file = glob.glob(path + f"bin_test_bin_size_{bin_size}.txt")
        if len(std_file)<1:
            f = open('/home/lakeclean/Documents/speciale/target_analysis/'
                     + f"bin_test_size_{bin_size}.txt",'w')
            f.write('ID, date, raw, no outliers, limited range 3800-6500Å\n')
            f.write(f'{epoch_name},{epoch_date},{raw_std},{no_out_std},{limited_std}\n')
        else:
            f = open('/home/lakeclean/Documents/speciale/target_analysis/'
                     + f"bin_test_size_{bin_size}.txt",'a')
            f.write('ID, date, raw, no outliers, limited range 3800-6500Å\n')
            f.write(f'{epoch_name},{epoch_date},{raw_std},{no_out_std},{limited_std}\n')
        f.close()


    


    return epoch_rv, epoch_vsini



#filename = '/home/lakeclean/Documents/speciale/initial_data/2024-04-02/FIHd020137_step011_merge.fits'

#analyse_spectrum(filename,use_SVD=False,show_bin_plots=False)

#### importing all the spectra:

lines = open('/home/lakeclean/Documents/speciale/merged_file_log.txt').read().split('\n')
files = []
IDs = []
dates = []
for line in lines[:-1]:
    line = line.split(',')
    file = line[2].strip()
    SEQID = line[1].strip()
    ID = line[0].strip()
    date = line[3].strip()
    if SEQID == 'science':
        files.append(file)
        IDs.append(ID)
        dates.append(date)


for file,ID,date in zip(files,IDs,dates):
    #analyse_spectrum(file,bin_size=200,use_SVD=,
    #                 show_bin_plots=False,show_plots=False)
    if ID == 'KIC-12317678':
        #if date == '2024-07-13T00:26:25.672':
            analyse_spectrum(file,bin_size=200,use_SVD=True,SB_type = 1,
                         show_bin_plots=False,save_data=False,
                         save_plots=False)
    
    













