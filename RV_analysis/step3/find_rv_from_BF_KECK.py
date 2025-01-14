import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd
from astropy.modeling import models, fitting
import sboppy as sb
from astropy.time import Time
import shazam
from convert_to_Campbell import find_Campbell
from astroquery.vizier import Vizier
from ophobningslov import *
import make_table_of_target_info as mt
import sys
master_path = '/usr/users/au662080'


def find_rv_time(ID,log_path,data_type,plotting=True):
    '''
    Function for plotting the radial velocity plot.
    Parameters:
        - ID             : [str] The ID of the target
        - log_path       : [str] The path to the log of the specific dataset we want to analyse
        - limits         : [list] first and last order of spectrum section
        

    Returns:
        prints resulting rv estimate for each epoch to file

    '''

    #The wavelength range that is "good":
    start = 3900 #Ångstrom
    end = 4845 #Ångstrom
    
    if True:
        path = log_path
        df = pd.read_csv(log_path)
        all_IDs = df['ID'].to_numpy()
        all_dates = df['date'].to_numpy()
        files = df['directory'].to_numpy()
        all_vbary = df['v_bary'].to_numpy()/1000
            
        def fit_line(x,y):
            fit = fitting.LinearLSQFitter()
            line_init = models.Linear1D()
            fitted_line = fit(line_init, x, y)
            slope = fitted_line.slope.value
            intercept = fitted_line.intercept.value
            new_x = np.linspace(min(x),max(x),10)
            return x,fitted_line(x),slope

        lines = open(f'{master_path}/Speciale/data/target_analysis/'+
                     f'{all_IDs[0]}/{all_dates[0][:len("2017-06-04")]}/data/bf_fit_params.txt').read().split('\n')[1:]
        print(f'Analyzing target: {ID}')
        print(f'Using the logfile: {log_path}')
        print(f'RV-data will be saved as: "{master_path}/Speciale/data/rv_data/{data_type}_{ID}.txt" ')

        epoch_rv1s = []
        epoch_rv2s = []
        epoch_rv1_errs = []
        epoch_rv2_errs = []
        epoch_jds = []
        epoch_dates = []
        epoch_vbary = []
        epoch_vsini1s = []
        epoch_vsini2s = []
        epoch_vsini1_errs = []
        epoch_vsini2_errs = []
        epoch_gwidths = []
        epoch_limbds = []
        epoch_ampl1s = []
        epoch_ampl2s = []
        epoch_consts = []

        ccd_order_offsets = {}
        #plotting all of the vrads - median(vrads) to see systematic offset
        fig, ax = plt.subplots()
        for i,all_ID in enumerate(all_IDs):
                
            date = all_dates[i]
            #print(date)
            path = f'{master_path}/Speciale/data/target_analysis/'
            path+= f'{all_IDs[i]}/{date[:len("2017-06-04")]}/data/bf_fit_params.txt'
            v_bary = all_vbary[i]
            
            #importing the values determined in analyse step
            df = pd.read_csv(path)
            bin_wls = df['bin_wls'].to_numpy()
            vrad1s = df['epoch_vrad1'].to_numpy()
            vrad2s = df['epoch_vrad2'].to_numpy()
            ccds = df['ccd'].to_numpy()
            orders = df['order'].to_numpy()


            #Index inside the specified good range
            idx = np.where( (bin_wls <end) & (bin_wls >start))[0]
            if len(idx)==0:
                print(bin_wls)
                input()
            



            #### Finding offset model of data: ####
            for j in range(len(bin_wls)):

                try: #Checking if ccd and order exist
                    #Adding to dictionary of ccd and orders
                    ccd_order_offsets[(ccds[j],orders[j])] = ccd_order_offsets[(ccds[j],orders[j])] +  [ vrad1s[j] - np.median(vrad1s[idx])]
                    if vrad1s[i] != vrad2s[i]:
                            ccd_order_offsets[(ccds[j],orders[j])] = ccd_order_offsets[(ccds[j],orders[j])] + [vrad2s[j] - np.median(vrad2s[idx])]
                except: #if ccd and order does not exist we create it
                    ccd_order_offsets[(ccds[j],orders[j])] = [vrad1s[j] - np.median(vrad1s[idx])]
                    if vrad1s[j] != vrad2s[j]:
                        ccd_order_offsets[(ccds[j],orders[j])] = ccd_order_offsets[(ccds[j],orders[j])] + [vrad2s[j] - np.median(vrad2s[idx])]

            #We plot the vrad to determine where the range is good:
            ax.scatter(bin_wls,vrad1s - np.median(vrad1s[idx]))
            ax.scatter(bin_wls,vrad2s - np.median(vrad2s[idx]))
            ax.vlines(start,-100,100,ls='--',color='b')
            ax.vlines(end,-100,100,ls='--',color='b')
            ax.set_xlabel('middle of wavelength range [Å]')
            ax.set_ylabel('Radial velocity [km/s]')
            ax.set_title(f'vrads offset by median of [{start},{end}] to show systematics')  

        if plotting: plt.show()
        plt.close()

        # Finding the proper median offset for each order and ccd
        ccd_order_weights = {}
        for i in ccd_order_offsets:
            length = len(ccd_order_offsets[i])
            if length == 1:
                ccd_order_weights[i] = sys.float_info.min
            else:
                ccd_order_weights[i] = 1/pow( np.std(ccd_order_offsets[i])/np.sqrt(len(ccd_order_offsets[i])) ,2) #following book
                if np.isnan(ccd_order_weights[i]):
                    print(ccd_order_offsets[i])
        
        #Plotting the individual vrad for each epoch to see where it is good and if the offset helps
        for i,all_ID in enumerate(all_IDs):
            
            if not ID == all_ID:
                continue
            
            date = all_dates[i]
            path = f'{master_path}/Speciale/data/target_analysis/'
            path+= f'{all_IDs[i]}/{date[:len("2017-06-04")]}/data/bf_fit_params.txt'
            jd = Time(date).jd #Correcting jd the way Frank likes. Something to with Tess.
            v_bary = all_vbary[i]
            epoch_jds.append(jd)


            #importing the values determined in analyse step
            df = pd.read_csv(path)
            vrad1s = df['epoch_vrad1'].to_numpy()
            vrad2s = df['epoch_vrad2'].to_numpy()
            ampl1s = df['epoch_ampl1'].to_numpy()
            ampl2s = df['epoch_ampl2'].to_numpy()
            vsini1s = df['epoch_vsini1'].to_numpy()
            vsini2s = df['epoch_vsini2'].to_numpy()
            gwidths = df['epoch_gwidth'].to_numpy()
            limbds = df['epoch_limbd'].to_numpy()
            consts = df['epoch_const'].to_numpy()
            ccds = df['ccd'].to_numpy()
            orders = df['order'].to_numpy()
            bin_wls = df['bin_wls'].to_numpy()

            weights = []
            for j in range(len(ccds)):
                weights.append(ccd_order_weights[(ccds[j],orders[j])])
            weights = np.array(weights)
                
            #Index inside the specified good range
            idx = np.where( (bin_wls <end) & (bin_wls >start))[0]


            #We plot the vrad to determine where the range is good:
            if False:
                fig, ax  = plt.subplots()
                ax.scatter(bin_wls,vrad1s,label='primary')
                ax.scatter(bin_wls,vrad2s,label='secondary')
                #ax.scatter(bin_wls,vrad1s,label='corrected primary')
                ax.vlines(start,-100,100,ls='--',color='b')
                ax.vlines(end,-100,100,ls='--',color='b')
                ax.set_xlabel('middle of wavelength range [Å]')
                ax.set_ylabel('Radial velocity [km/s]')
                ax.set_title(f'ID: {ID}, date: {date}')
                ax.legend()
                plt.show()
            
            #We choose the vsini as the mean of the order vsini's that lie in good range:
            mean_vsini1 = np.mean(vsini1s[idx])
            err_vsini1 = np.std(vsini1s[idx])/len(vsini1s[idx])
            mean_vsini2 = np.mean(vsini2s[idx])
            err_vsini2 = np.std(vsini2s[idx])/len(vsini2s[idx])
            
            epoch_vsini1s.append(mean_vsini1)
            epoch_vsini2s.append(mean_vsini2)
            epoch_vsini1_errs.append(err_vsini1)
            epoch_vsini2_errs.append(err_vsini2)

            epoch_gwidths.append(np.median(gwidths[idx]))
            epoch_limbds.append(np.median(limbds[idx]))
            epoch_ampl1s.append(np.median(ampl1s[idx]))
            epoch_ampl2s.append(np.median(ampl2s[idx]))
            epoch_consts.append(np.median(consts[idx]))
            
            epoch_dates.append(date)
            epoch_vbary.append(v_bary)


            #Selecting the best region of orders
            vrad1s = vrad1s[idx]
            vrad2s = vrad2s[idx]
            weights = weights[idx]

            #normalized weights
            norm_weights = weights / sum(weights)
           
            #Weighted mean
            epoch_rv1s.append(sum(vrad1s*norm_weights))
            epoch_rv2s.append(sum(vrad2s*norm_weights))

            #standard err of weighted mean from unc propagation        
            epoch_rv1_errs.append(1/np.sqrt(sum(weights)))
            epoch_rv2_errs.append(1/np.sqrt(sum(weights)))
            


        epoch_rv1s = np.array(epoch_rv1s)
        epoch_rv2s = np.array(epoch_rv2s)
        epoch_rv1_errs = np.array(epoch_rv1_errs)
        epoch_rv2_errs = np.array(epoch_rv2_errs)
        epoch_jds = np.array(epoch_jds)
        epoch_dates = np.array(epoch_dates)
        epoch_vbary = np.array(epoch_vbary)
        epoch_vsini1s = np.array(epoch_vsini1s)
        epoch_vsini2s = np.array(epoch_vsini2s)
        epoch_vsini1_errs = np.array(epoch_vsini1_errs)
        epoch_vsini2_errs = np.array(epoch_vsini2_errs)
        epoch_gwidths = np.array(epoch_gwidths)
        epoch_limbds = np.array(epoch_limbds)
        epoch_ampl1s = np.array(epoch_ampl1s)
        epoch_ampl2s = np.array(epoch_ampl2s)
        epoch_consts = np.array(epoch_consts)
        

        #Printing rv_estimations to file:
        rv_path = f'{master_path}/Speciale/data/rv_data/{data_type}_{ID}.txt'
        rv_names = ['jd','date','rv1','rv2','e_rv1','e_rv2','vbary',
                    'vsini1','vsini2','e_vsini1','e_vsini2','ampl1','ampl2', 'const', 'gwidth', 'limbd']
        
        rv_dat = [epoch_jds,epoch_dates,epoch_rv1s,epoch_rv2s,
                  epoch_rv1_errs,epoch_rv2_errs,epoch_vbary,
                  epoch_vsini1s, epoch_vsini2s,epoch_vsini1_errs, epoch_vsini2_errs,
                  epoch_ampl1s, epoch_ampl2s, epoch_consts, epoch_gwidths, epoch_limbds]
        rv_dict = {}
        for i in range(len(rv_names)):
            rv_dict[rv_names[i]] = rv_dat[i]
        df = pd.DataFrame(rv_dict)
        #Only making a file if data is not empty
        #print(rv_dat)
        if len(rv_dat[0])>0:
            df.to_csv(rv_path,index=False)

            

data_types = ['KECK']
log_paths = [f'{master_path}/Speciale/data/KECK_order_file_log.txt']

for data_type, log_path in zip(data_types, log_paths):

    #KIC9693187
    if True:
        find_rv_time('KIC9693187',
                     log_path=log_path,
                     data_type=data_type)
   

    #KIC10454113
    if True:
        find_rv_time('KIC10454113',
                     log_path=log_path,
                     data_type=data_type)























    
