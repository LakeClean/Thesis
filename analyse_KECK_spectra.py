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
from PyAstronomy import pyasl #For converting from vacuum to air



#### importing templates ####

#RGB from ardata
template_dir = '/home/lakeclean/Documents/speciale/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
arcturus_tfl_RG = template_data['arcturus']
arcturus_twl = template_data['wavelength']
tfl_MS = template_data['solarflux']



#import the global info
tab = mt.get_table()
all_target_names = tab['ID'].data # Just a list of the names
#RGBs = tab['star_type'].data # names of known RGB stars
#Teffs = tab['Teff'].data #Effective temperature [K]
#loggs = tab['loggs'].data
#FeHs = tab['Fe/H'].data

def analyse_spectrum(folder,target_name, template='MS',start_order=1, append_to_log=False,
                     
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
    Analyses a raw ordered spectra.
    The function needs the path to the ordered fits file.
    It then stores the analysis of the spectra in existing directory.
    
    :params:
        file          :str, path of the fits file
        start_wl      :int, starting wavelength [Ångstrom]
        end_wl        :int, ending wavelength [Ångstrom]
        template      :str, either 'MS' or 'RG'
        bin_size      :int, size of wavelength bin
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
    
    #Importing data for the specific epoch:
    files = glob.glob(f'{folder}/*')
    def file_sorter(file):
        ccd = int(file.split('_')[2])
        order = int(file.split('_')[3])
        return (ccd,order)
    
    files.sort(key=file_sorter)

    no_orders = len(files)

    epoch_name = target_name #name of target
    epoch_date = folder[-10:]  #date of fits creation
    print(epoch_name, epoch_date)
    #Path for the reduced data:
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
        
    def correct_logg(x):
        test = x%0.5
        if test >=0.25:
            return x + 0.5 - test
        else:
            return x -test
    ########################################################################


    #Selcting the appropiate template:
    if mt.get_value('star_type',target_name) == 'RGB':
        tfl = arcturus_tfl_RG
        twl = arcturs_twl
    else:
        Teff = int(np.round(float(mt.get_value('Teff',target_name)),-2))
        logg = correct_logg(float(mt.get_value('logg',target_name)))
        FeH = mt.get_value('Fe/H',target_name)
        templates_path = '/home/lakeclean/Documents/speciale/templates/phoenix/*'
        tfl = tfl_MS
        twl = arcturus_twl
        

        
    #Raw spectrum is plotted
    fig, ax = plt.subplots()
    for order in files:
        
        x_label, y_label, title ='wavelength [Å]', 'flux (raw)', 'ordered'
        z = pyfits.getdata(order)
        header = pyfits.getheader(order)
        #print(header['HELIOVEL'],header['DATE'])
        xs, ys = z['wave'],z['flux']
        xs = pyasl.vactoair2(xs) #Converting to air wavelength

        ax.plot(xs,ys)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plot_title = title #+ ' ' + epoch_name + ' ' + epoch_date
        ax.set_title(plot_title+ ' ' + epoch_name + ' ' + epoch_date)
        #ax.set_ylim(np.percentile(ys,99)*2,np.percentile(ys,1)*2)
    
    if save_plots: fig.savefig(path+f"/plots/{plot_title.replace(' ','_')}.svg",
                                   dpi='figure', format='svg')
    
    if save_data: save_datas([xs,ys],[x_label,y_label],plot_title.replace(' ','_'))
    if show_plots: plt.show()
    plt.close()
    
    

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

    epoch_ampl1, epoch_gwidth = np.zeros(no_orders), np.zeros(no_orders) #parameters of fit to bf
    epoch_vrad1, epoch_vsini1  = np.zeros(no_orders), np.zeros(no_orders)#parameters of fit to bf
    epoch_limbd, epoch_const = np.zeros(no_orders), np.zeros(no_orders)#parameters of fit to bf

    epoch_ampl2 = np.zeros(no_orders)#parameters of fit to bf
    epoch_vrad2, epoch_vsini2  = np.zeros(no_orders), np.zeros(no_orders)#parameters of fit to bf

    slopes = np.zeros(no_orders) 
    bin_wls = np.zeros(no_orders)
    ccds = np.zeros(no_orders) #KECK data has inconsistent amount of orders
    orders = np.zeros(no_orders) 

    for i in np.arange(0,no_orders,1):

        #find the order of the file:
        ccd = int(files[i].split('_')[2])
        order = int(files[i].split('_')[3])
        ccds[i] = ccd
        orders[i] = order

        
        #Pick out correct wl range
        z = pyfits.getdata(files[i]) 
        wl, fl = z['wave'],z['flux']
        wl = pyasl.vactoair2(wl) #converting from vacuum to air
        
        bin_wls[i] = wl[int(len(wl)/2)]

        #Check if template does not cover spectrum:
        if wl[0] < twl[0]:
            print(f'ccd {ccd}, order {order} was outside template:', wl[0], twl[0])
            continue
        
        
        if save_data: save_datas([wl,fl],['wavelength [Å]', 'flux (raw)'],
                                 f"order_{order}_ccd_{ccd}_raw_spectrum")
        
        if np.mean(fl)<0.001: print('flux is very low') 

        #normalize
        nwl, nfl = shazam.normalize(wl,fl, normalize_bl,normalize_poly,
                                    normalize_gauss,normalize_lower,normalize_upper)

        nfl = nfl / np.median(nfl[np.where(np.percentile(nfl,95)<nfl)[0]])#The flux of the normalized flux that is above 95% percentile
        
        if save_data: save_datas([nwl,nfl],['wavelength [Å]', 'flux (norm)'],
                                 f"order_{order}_ccd_{ccd}_normalized")
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
        #print(nwl,nfl,twl,tfl)
        r_wl, rf_fl, rf_tl = shazam.resample(nwl,nfl,twl,tfl, resample_dv, resample_edge)
        if save_data: save_datas([r_wl, rf_fl, rf_tl],
                                 ['wavelength(resampled) [Å]',
                                  'flux (resampled and flipped)',
                                  'template flux (resampled and flipped)']
                                  ,f"order_{order}_ccd_{ccd}_resampled_flipped")
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
                                     f"order_{order}_ccd_{ccd}_broadening_function")
        
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

            fig.suptitle(f'Order #{i} ' + epoch_name + ' ' + epoch_date)
            
            
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
            
    if save_data: save_datas([bin_wls, epoch_vrad1, epoch_vrad2,epoch_ampl1, epoch_ampl2,
                                  epoch_vsini1, epoch_vsini2, epoch_gwidth, epoch_limbd,
                                  epoch_const,ccds,orders],
                                     ['bin_wls', 'epoch_vrad1', 'epoch_vrad2','epoch_ampl1', 'epoch_ampl2',
                                  'epoch_vsini1', 'epoch_vsini2', 'epoch_gwidth', 'epoch_limbd',
                                  'epoch_const','ccd','order'],
                                     f"bf_fit_params")

    # Plotting bfs or ccfs together:
    if use_SVD:
        fig, ax = plt.subplots()
        x_label, y_label, title ='rvs [km/s]', 'broadening function', 'broadening function'
        k=0
        for xs, ys,smoothed,model in zip(epoch_rvs,epoch_bf,epoch_smoothed_bf,epoch_model):
            #ax.plot(xs,ys+k)
            ax.plot(xs,smoothed+k,label='smoothed bf')
            ax.plot(xs,model+k,label='fit to smoothed bf')
            k+=0.1
        #ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0,no_orders*0.1)
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
        ax.set_ylim(0,no_orders)
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
    x_label, y_label='order_wl', 'Radial Velocity 1 [km/s]'
    title = 'radial velocity per order Uncorrected'
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
    x_label, y_label='order_wl', 'Vsini[km/s]'
    title = 'vsini per order'
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
'''
lines = open('/home/lakeclean/Documents/speciale/NOT_order_file_log.txt').read().split('\n')
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
'''
SB2IDs =['KIC-9693187','KIC-9025370','KIC9652971']
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
            
        



k=0
time1 = time()


folder_paths = glob.glob('/home/lakeclean/Documents/speciale/initial_data/KECK/KIC10454113/*')
folder_paths.sort()


for folder in folder_paths:
    #date = folder.split('/')[-1]
    
    #if ID == 'KIC12317678':
    #if date == '2017-07-08':
        #if Time(date).jd > 2460640:
            #print(f'Spectrum: {k}/{len(files)}, Time: {time()-time1}s')
            time1 = time()
            k+=1
            start_order=1
            show_bin_plots=False
            save_data=True
            save_plots=False
            show_plots=False
            rotbf_fit_print_report=False
            target_name = 'KIC10454113'
            analyse_spectrum(folder,target_name =target_name, SB_type = 1, start_order=start_order,
                             show_bin_plots=show_bin_plots,save_data=save_data,
                             save_plots=save_plots,show_plots=show_plots,
                             rotbf_fit_print_report=rotbf_fit_print_report)

    
    





