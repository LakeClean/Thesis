import shazam
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import glob
#import pickle #For saving figures as matplotlib figures
from astropy.modeling import models, fitting
from time import time
from scipy.signal import find_peaks
from astropy.time import Time
import make_table_of_target_info as mt
import read_goettingen as gg

#import template from arcturus
template_dir = '/home/lakeclean/Documents/speciale/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
tfl_RG = template_data['arcturus']
#tfl_MS = template_data['solarflux']
twl = template_data['wavelength']

#import template from goettingen:
_a, tfl_gg, _b, twl_gg = gg.get_goettingen()


#fig, ax  = plt.subplots()
#ax.plot(twl,tfl_MS)
#plt.show()

#import the RGB info
tab = mt.get_table()
all_target_names = tab['ID'].data # Just a list of the names
RGBs = tab['star_type'].data # names of known RGB stars
all_ras = tab['RA'].data # right ascension of all stars
all_decs = tab['DEC'].data #declination of all stars


def analyse_spectrum(file, template='MS',start_wl=0, end_wl=100000,
                     bin_size=200, append_to_log=False,
                     
                     normalize_bl = np.array([]),normalize_poly=1,normalize_gauss=True,
                     normalize_lower=0.5,normalize_upper=1.5,
                     
                     crm_iters = 1, crm_q = [99.0,99.9,99.99],
                     
                     resample_dv=1.0, resample_edge=0.0,
                     
                     getCCF_rvr=401, getCCF_ccf_mode='full',
                     
                     getBF_rvr=401, getBF_dv=1.0,
                     
                     rotbf2_fit_fitsize=30,rotbf2_fit_res=60000,rotbf2_fit_smooth=2,
                     rotbf2_fit_vsini1=10.0,rotbf2_fit_vsini2=5.0,rotbf2_fit_vrad1=-30.0,
                     rotbf2_fit_vrad2=10.0,rotbf2_fit_ampl1=0.05,rotbf2_fit_ampl2=0.05,
                     rotbf2_fit_print_report=False,rotbf2_fit_smoothing=True,
                     
                     rotbf_fit_fitsize=30,rotbf_fit_res=60000,rotbf_fit_smooth=2.0,
                     rotbf_fit_vsini=10.0,rotbf_fit_print_report=True,
                     
                     use_SVD=True,SB_type=1,
                     show_plots=True, save_plots=True, save_data = True,
                     show_bin_plots=False,save_bin_info=False): 
    '''
    Analyses a raw merged spectra.
    The function needs the path to the ordered fits file.
    It then stores the analysis of the spectra in existing directory.
    
    :params:
        file          :str, path of the fits file
        start_wl      :int, starting wavelength [Ångstrom]
        end_wl        :int, ending wavelength [Ångstrom]
        template      :str, either 'MS' or 'RG'
        bin_size      :int, size of wavelength bin in Å
        SB_type       :int, either 1 or 2 works atm. when use_SVD=True
        append_to_log :bool, append the given parameters to a log

        rvr           : integer, range for RVs in km/s
        fitsize       : float, fitsize in km/s - only fit `fitsize` part of the rotational profile
        res           : float, resolution of spectrograph - affects width of peak
        smooth        : float, smoothing factor - sigma in Gaussian smoothing
        
    
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
        - for loop for every order
            - Select the right order
            - normalize spectrum with shazamm.normalize
            - pick out 95% percentile and further normalize with this
            - remove cosmic rays with shazam.crm
            - fit line to normalized spectrum to see if flat
            - resample and flip spectrum and template with shazam.resample
            -
    '''
    #######################################################################
    def save_datas(datas,labels,title):
        '''
        Function for saving the data:
         - datas: list of lists
         - labels: labels of the input of datas list
         - title: The title of file data is saved to
        '''
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

    def sort_spectrum(wl,fl):
        '''
        Function for sorting illsorted spectra:
        '''
        new_fl = [x for y,x in sorted(zip(wl,fl))]
        new_wl = [y for y,x in sorted(zip(wl,fl))]
        return np.array(new_wl),np.array(new_fl)
    ########################################################################

        
    fits_data = pyfits.getdata(file) #flux info
    header = pyfits.getheader(file)

    #We have to start from the starting wavelength
    #lam = header['CDELT1']*np.arange(header['NAXIS1'])+header['CRVAL1'] #wavelengths
    lam = fits_data[3,:]*10 #The unnormlaized wavelength (is in nm and is therefore changed to Å
    data = fits_data[4,:] #The 'Intensity' or flux of the spectra

    lam,data = sort_spectrum(lam,data) #spectrum is sorted
    
    if start_wl<lam[0]:
        start_wl = lam[0]
    if end_wl > lam[-1]:
        end_wl = lam[-1]
    
    #Finding out the name of target based on RA and DEC:
    ra = header['RA_DEG'] #The right ascension of target
    dec = header['DEC_DEG'] #The declination of target
    coord_dist = np.sqrt( (all_ras- ra)**2 + (all_decs-dec)**2)
    epoch_name = all_target_names[np.where(coord_dist == min(coord_dist))[0]][0]

    epoch_date = header['DATE'].strip(' ')
    
    #epoch_Vhelio = header['VHELIO'] # heliocentric velocity
    #epoch_name = header['HIERARCH TNG OBS TARG NAME'].strip(' ').replace('-','') #name of target
    print(epoch_name, epoch_date)
    
    if epoch_name in RGBs:
        tfl = tfl_RG
    else:
        tfl = tfl_MS

    path = '/home/lakeclean/Documents/speciale/target_analysis/' + epoch_name +'/' + epoch_date

    #######################################################################

    if append_to_log:
        f = open('analyse_log.txt').read()
        f += f'{start_order}, {append_to_log}, {normalize_bl}, {normalize_poly}, {normalize_gauss},'
        f += f'{normalize_lower}, {normalize_upper}, {crm_iters}, {crm_q}, {resample_dv}, {resample_edge},'
        f += f' {getCCF_rvr}, {getCCF_ccf_mode}, {getBF_rvr}, {getBF_dv},'
        f += f'{rotbf2_fit_fitsize},{rotbf2_fit_res},{rotbf2_fit_smooth},'
        f += f'{rotbf2_fit_vsini1},{rotbf2_fit_vsini2},{rotbf2_fit_vrad1},'
        f += f'{rotbf2_fit_vrad2},{rotbf2_fit_ampl1},{rotbf2_fit_ampl2},'
        f += f'{rotbf2_fit_print_report},{rotbf2_fit_smoothing},'
        f += f'{rotbf_fit_fitsize},{rotbf_fit_res},{rotbf_fit_smooth},'
        f += f'{rotbf_fit_vsini},{rotbf_fit_print_report},'
        f += f'{use_SVD},{SB_type},{show_plots}, {save_plots}, {save_data},'
        f += f'{show_bin_plots},{save_bin_info}'
        lines = f
        f.close()
        f.open('analyse_log.txt','w')
        f.write(lines)
        f.close()
    #########################################################################


    
        
    #Raw spectrum is plotted
    
    fig, ax = plt.subplots()
    x_label, y_label, title ='wavelength [Å]', 'flux (raw)', 'ordered'
    xs, ys = lam, data
    ax.plot(xs,ys)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
    ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
    
    if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                                   dpi='figure', format='svg')
    
    if save_data: save_datas([xs,ys],[x_label,y_label],plot_title.replace(' ','_'))
    if show_plots: plt.show()
    plt.close()
    
    
    n_bins = int((end_wl-start_wl)//bin_size) #number of bins is found
    
    epoch_nwls = [] #binned norm wl 
    epoch_nfls = [] #binned norm fl

    epoch_rf_wls = [] #binned resamp and flipped wl 
    epoch_rf_fls = [] #binned resamp and flipped fl
    epoch_rf_tfls = [] #binned resamp and flipped template

    epoch_rvs = [] #binned rvs for BF or ccf
    epoch_bf = [] #binned BF
    epoch_model = [] # The fit to the bf
    epoch_smoothed_bf = [] #binned smoothed BF

    epoch_ccf = [] #binned BF

    epoch_ampl1, epoch_gwidth = np.zeros(n_bins), np.zeros(n_bins) #parameters of fit to bf
    epoch_vrad1, epoch_vsini1  = np.zeros(n_bins), np.zeros(n_bins)#parameters of fit to bf
    epoch_limbd, epoch_const = np.zeros(n_bins), np.zeros(n_bins)#parameters of fit to bf

    epoch_ampl2 = np.zeros(n_bins)#parameters of fit to bf
    epoch_vrad2, epoch_vsini2  = np.zeros(n_bins), np.zeros(n_bins)#parameters of fit to bf

    slopes = np.zeros(n_bins) 
    #bin_wls = np.zeros(n_bins)
    bin_wls = np.arange(n_bins)* bin_size +start_wl

    for i in np.arange(n_bins):
        begin = start_wl + bin_size*i
        end = begin + bin_size

        #Pick out correct wl range
        index = np.where((lam>begin) & ( lam<end))[0]
        wl = lam[index]
        fl = data[index]

        #Check if template does not cover spectrum:
        if (wl[0] < twl[0]) or (wl[-1]>twl[-1]):
            print(f'order {i}[{wl[0]},{wl[-1]}] was outside template[{twl[0]},{twl[-1]}]')
            continue
        
        if save_data: save_datas([wl,fl],['wavelength [Å]', 'flux (raw)'],
                                 f"order_{i}_raw_spectrum")
        
        if np.mean(fl)<0.001: print('flux is very low') 

        #normalize
        nwl, nfl = shazam.normalize(wl,fl, normalize_bl,normalize_poly,
                                    normalize_gauss,normalize_lower,normalize_upper)

        nfl = nfl / np.median(nfl[np.where(np.percentile(nfl,95)<nfl)[0]])#The flux of the normalized flux that is above 95% percentile
        
        if save_data: save_datas([nwl,nfl],['wavelength [Å]', 'flux (norm)'],
                                 f"order_{i}_normalized")
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
                                  ,f"order_{i}_resampled_flipped")
        epoch_rf_wls.append(r_wl)
        epoch_rf_fls.append(rf_fl)
        epoch_rf_tfls.append(rf_tl)
        

        #If we want to get BF from SVD
        if use_SVD:
            
            rvs,bf = shazam.getBF(rf_fl,rf_tl,getBF_rvr, getBF_dv)
            epoch_rvs.append(rvs)
            epoch_bf.append(bf)

            #Fitting bf:
            if SB_type == 1:
                fit, model, bfgs = shazam.rotbf_fit(rvs,bf,rotbf_fit_fitsize,rotbf_fit_res,
                                                    rotbf_fit_smooth,rotbf_fit_vsini,
                                                    rotbf_fit_print_report)
                #rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value

                epoch_ampl1[i], epoch_gwidth[i] = fit.params['ampl1'].value, fit.params['gwidth'].value
                epoch_vrad1[i], epoch_vsini1[i]  = fit.params['vrad1'].value, fit.params['vsini1'].value
                epoch_limbd[i], epoch_const[i] = fit.params['limbd1'].value, fit.params['const'].value
                epoch_vrad2[i], epoch_vsini2[i] = fit.params['vrad1'].value, fit.params['vsini1'].value
                epoch_ampl2[i] = fit.params['ampl1'].value
                #print(epoch_vsini1[i],epoch_gwidth[i])
                
            elif SB_type == 2:
                fit, model, bfgs = shazam.rotbf2_fit(rvs,bf, rotbf2_fit_fitsize,rotbf2_fit_res,
                                                     rotbf2_fit_smooth, rotbf2_fit_vsini1,
                                                     rotbf2_fit_vsini2,rotbf2_fit_vrad1,
                                                     rotbf2_fit_vrad2,rotbf2_fit_ampl1,
                                                     rotbf2_fit_ampl2,rotbf2_fit_print_report,
                                                     rotbf2_fit_smoothing)
                
                #rv, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
                
                epoch_ampl1[i], epoch_ampl2[i] = fit.params['ampl1'].value,fit.params['ampl2'].value
                epoch_vrad1[i], epoch_vsini1[i] = fit.params['vrad1'].value, fit.params['vsini1'].value
                epoch_vrad2[i], epoch_vsini2[i] = fit.params['vrad2'].value, fit.params['vsini2'].value
                epoch_limbd[i], epoch_const[i] = fit.params['limbd1'].value, fit.params['const'].value
                epoch_gwidth[i] = fit.params['gwidth'].value
            
            else:
                print('The SB type was given wrong: Should be int 1 or 2')

                
            
            epoch_smoothed_bf.append(bfgs)
            epoch_model.append(model)

            
            if save_data: save_datas([rvs,bf,bfgs,model],
                                     ['wavelength [km/s]', 'broadening function',
                                      'smoothed_bf', 'Fitted model'],
                                     f"order_{i}_broadening_function")
        
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
            ax.plot(rvs,bf,label='Broadening Function')
            ax.plot(rvs,bfgs,label='Smoothed broadening Function')
            ax.plot(rvs,model,label='Fit to BF')
            ax.set_xlabel('radial velocity [km/s]')
            ax.set_ylabel('Broadening Function')


            if epoch_ampl1[i] < epoch_ampl2[i]:
                ax.scatter(epoch_vrad1[i],0,color='b',label=f'{epoch_ampl1[i]}')
                ax.scatter(epoch_vrad2[i],0,color='r',label=f'{epoch_ampl2[i]}')
                
            if epoch_ampl1[i] > epoch_ampl2[i]:
                ax.scatter(epoch_vrad1[i],0,color='r',label=f'{epoch_ampl1[i]}')
                ax.scatter(epoch_vrad2[i],0,color='b',label=f'{epoch_ampl2[i]}')
                
            ax.legend()
            plt.show()
            plt.close()
            
    if save_data: save_datas([epoch_vrad1, epoch_vrad2,epoch_ampl1, epoch_ampl2,
                                  epoch_vsini1, epoch_vsini2, epoch_gwidth, epoch_limbd,
                                  epoch_const],
                                     ['epoch_vrad1', 'epoch_vrad2','epoch_ampl1', 'epoch_ampl2',
                                  'epoch_vsini1', 'epoch_vsini2', 'epoch_gwidth', 'epoch_limbd',
                                  'epoch_const'],
                                     f"bf_fit_params")

    # Plotting bfs together:
    fig, ax = plt.subplots()
    x_label, y_label, title ='rvs [km/s]', 'broadening function', 'broadening function'
    k=0
    for xs, ys,smoothed,model in zip(epoch_rvs,epoch_bf,epoch_smoothed_bf,epoch_model):
        ax.plot(xs,smoothed+k,label='smoothed bf')
        ax.plot(xs,model+k,label='fit to smoothed bf')
        k+=0.1
    #ax.legend()
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


    
    # Plotting radial velocities measured:
    fig, ax = plt.subplots()
    x_label, y_label='bin_wl', 'Radial Velocity 1 [km/s]'
    title = 'radial velocity per bin Uncorrected'
    xs, ys = bin_wls,epoch_vrad1
    ax.scatter(xs,ys,label='rv1')
    ax.scatter(bin_wls,epoch_vrad2,label='rv2')
    ax.plot([xs[0],xs[-1]],[np.mean(ys),np.mean(ys)],label='mean',color='r')
    #ax.plot([xs[0],xs[-1]],[epoch_Vhelio,epoch_Vhelio],label=f'V_helio={epoch_Vhelio}',color='g')
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
    ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
    if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                               dpi='figure', format='svg')
    if show_plots: plt.show()
    plt.close()
    '''
    
    if save_data: save_datas([bin_wls,epoch_vrad1,epoch_vrad2],[x_label,y_label,'Radial Velocity 1 [km/s]'],plot_title.replace(' ','_'))
    plt.close()

    '''
    # Plotting vsini measured:
    fig, ax = plt.subplots()
    x_label, y_label='bin_wl', 'Vsini[km/s]'
    title = 'vsini per bin'
    xs, ys = bin_wls,epoch_vsini1
    ax.scatter(xs,ys,label='vsini')
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
    
    

    return epoch_vrad1, epoch_vsini1



#filename = '/home/lakeclean/Documents/speciale/initial_data/2024-04-02/FIHd020137_step011_merge.fits'

#analyse_spectrum(filename,use_SVD=False,show_bin_plots=False)

#### importing all the spectra:

lines = open('/home/lakeclean/Documents/speciale/ESpaDOns_merged_file_log.txt').read().split('\n')
files = []
IDs = []
dates = []
for line in lines[1:-1]:
    line = line.split(',')
    file = line[1].strip()
    #SEQID = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)

SB2IDs =['KIC9693187','KIC9025370','KIC9652971']
IDlines = open('/home/lakeclean/Documents/speciale/spectra_log_h_readable.txt').read().split('&')
SB2_IDs, SB2_dates, SB2_types, vguess1s, vguess2s = [], [], [], [], []
for IDline in IDlines[:-1]:
    if IDline.split(',')[0][11:].strip(' ') in SB2IDs:
        for line in IDline.split('\n')[2:-1]:
            line = line.split(',')

            if line[2].split('/')[0].strip(' ') == 'NaN':
                continue
            SB2_IDs.append(IDline.split(',')[0][11:].strip(' '))
            SB2_dates.append(line[0].strip(' '))
            SB2_types.append(line[1].strip(' '))
            vguess1s.append(line[2].split('/')[0].strip(' '))
            vguess2s.append(line[2].split('/')[1].strip(' '))
            
        
#Typical resolutions: NOT (HIRES = 67000, LOWRES = 25000) https://www.not.iac.es/instruments/fies/
#                     TNG (HARPS = 115000)
#                     KECK (HIRES = 47700)
#                     CFHT: Apparantly all over the place: (65000-90000)


k=0
time1 = time()
for file,ID,date in zip(files[0:],IDs[0:],dates[0:]):

    if ID == 'KIC9025370':
            
        #if date == '2024-04-01T23:36:23.915':
        #if Time(date).jd > 2460618.4256590856:
            print(f'Spectrum: {k}/{len(files)}, Time: {time()-time1}s')
            time1 = time()
            k+=1
            show_bin_plots=False
            save_data=False
            save_plots=False
            show_plots=True
            rotbf_fit_print_report=False
            start_wl=00
            end_wl=100000
            bin_size=80
            resolution = 70000#Notice different instruments have different resolution

            
            if ID not in SB2IDs:
                    analyse_spectrum(file,SB_type = 1, bin_size=bin_size, start_wl = start_wl, end_wl=end_wl,
                             show_bin_plots=show_bin_plots,save_data=save_data,
                             save_plots=save_plots,show_plots=show_plots,
                             rotbf_fit_print_report=rotbf_fit_print_report,
                             rotbf_fit_res=resolution)

            else:
                for SB2_date,SB2_type,vguess1,vguess2 in zip(SB2_dates,SB2_types, vguess1s,vguess2s):
                    if SB2_date == date:
                        if SB2_type == '1':
                            analyse_spectrum(file,SB_type = 1, bin_size=bin_size, start_wl = start_wl, end_wl=end_wl,
                                     show_bin_plots=show_bin_plots,save_data=save_data,
                                     save_plots=save_plots,show_plots=show_plots,
                                     rotbf2_fit_print_report=rotbf_fit_print_report,
                                     rotbf_fit_res=resolution)
                                
                        if SB2_type == '2':
                            print('Initial guesses: ', vguess1,vguess2)
                            analyse_spectrum(file,SB_type = 2, bin_size=bin_size, start_wl = start_wl, end_wl=end_wl,
                                     show_bin_plots=show_bin_plots,save_data=save_data,
                                     save_plots=save_plots,show_plots=show_plots,
                                     rotbf2_fit_print_report=rotbf_fit_print_report,
                                     rotbf2_fit_vrad1=float(vguess1),
                                     rotbf2_fit_vrad2=float(vguess2),
                                     rotbf2_fit_res=resolution)

                            



                             #rotbf_fit_fitsize=30,rotbf_fit_res=60000,rotbf_fit_smooth=2.0,
                             #rotbf_fit_vsini=5.0,rotbf_fit_print_report=True)
    
    

'''
normalize_bl = np.array([]),normalize_poly=1,normalize_gauss=True,
                     normalize_lower=0.5,normalize_upper=1.5,
                     
                     crm_iters = 1, crm_q = [99.0,99.9,99.99],
                     
                     resample_dv=1.0, resample_edge=0.0,
                     
                     getCCF_rvr=401, getCCF_ccf_mode='full',
                     
                     getBF_rvr=401, getBF_dv=1.0,
                     
                     rotbf2_fit_fitsize=30,rotbf2_fit_res=60000,rotbf2_fit_smooth=2.0,
                     rotbf2_fit_vsini1=10.0,rotbf2_fit_vsini2=5.0,rotbf2_fit_vrad1=-30.0,
                     rotbf2_fit_vrad2=-17.0,rotbf2_fit_ampl1=0.5,rotbf2_fit_ampl2=0.5,
                     rotbf2_fit_print_report=False,rotbf2_fit_smoothing=True,
                     
                     rotbf_fit_fitsize=30,rotbf_fit_res=60000,rotbf_fit_smooth=2.0,
                     rotbf_fit_vsini=5.0,rotbf_fit_print_report=True,
                     
                     use_SVD=False,SB_type=1,
                     show_plots=True, save_plots=True, save_data = True,
                     show_bin_plots=False,save_bin_info=False):
'''




