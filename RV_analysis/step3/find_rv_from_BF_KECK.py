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


#f'/home/lakeclean/Documents/speciale/TNG_merged_file_log.txt'
def find_rv_time(ID,log_path,data_type,plotting=True,ask_for_limits=True):
    '''
    Function for plotting the radial velocity plot.
    Parameters:
        - ID             : [str] The ID of the target
        - log_path       : [str] The path to the log of the specific dataset we want to analyse
        - limits         : [list] first and last order of spectrum section
        

    Returns:
        prints resulting rv estimate for each epoch to file

    '''
    
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

        lines = open(f'/home/lakeclean/Documents/speciale/target_analysis/'+
                     f'{all_IDs[0]}/{all_dates[0][:len("2017-06-04")]}/data/bf_fit_params.txt').read().split('\n')[1:]
        print(f'Analyzing target: {ID}')
        print(f'Using the logfile: {log_path}')
        print(f'RV-data will be saved as: "/home/lakeclean/Documents/speciale/rv_data/{data_type}_{ID}.txt" ')

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
            print(date)
            path = f'/home/lakeclean/Documents/speciale/target_analysis/'
            path+= f'{all_IDs[i]}/{date[:len("2017-06-04")]}/data/bf_fit_params.txt'
            v_bary = all_vbary[i]
            
            #importing the values determined in analyse step
            df = pd.read_csv(path)
            bin_wls = df['bin_wls'].to_numpy()
            vrad1s = df['epoch_vrad1'].to_numpy() + v_bary
            vrad2s = df['epoch_vrad2'].to_numpy() + v_bary
            ccds = df['ccd'].to_numpy()
            orders = df['order'].to_numpy()

            #### Finding offset model of data: ####
            for i in range(len(bin_wls)):
                try: #Adding to dictionary of ccd and orders
                    ccd_order_offsets[(ccds[i],orders[i])] = ccd_order_offsets[(ccds[i],orders[i])] +  [ vrad1s[i] - np.median(vrad1s)]
                    if vrad1s[i] != vrad2s[i]:
                            ccd_order_offsets[(ccds[i],orders[i])] = ccd_order_offsets[(ccds[i],orders[i])] + [vrad2s[i] - np.median(vrad2s)]
                except:
                    ccd_order_offsets[(ccds[i],orders[i])] = [vrad1s[i] - np.median(vrad1s)]
                    if vrad1s[i] != vrad2s[i]:
                        ccd_order_offsets[(ccds[i],orders[i])] = ccd_order_offsets[(ccds[i],orders[i])] + [vrad2s[i] - np.median(vrad2s)]

            #We plot the vrad to determine where the range is good:
            ax.scatter(bin_wls,vrad1s - np.median(vrad1s[10:30]))
            ax.scatter(bin_wls,vrad2s - np.median(vrad2s[10:30]))
            ax.set_xlabel('middle of wavelength range [Å]')
            ax.set_ylabel('Radial velocity [km/s]')
            ax.set_title(f'vrads offset by median to show systematics')  

        if plotting: plt.show()
        plt.close()

        # Finding the proper median offset for each order and ccd
        ccd_order_weights = {}
        for i in ccd_order_offsets:
            ccd_order_weights[i] = np.std(ccd_order_offsets[i])
            ccd_order_offsets[i] = np.median(ccd_order_offsets[i])
            
        
        #Plotting the individual vrad for each epoch to see where it is good and if the offset helps
        for i,all_ID in enumerate(all_IDs):
            
            if not ID == all_ID:
                continue
            
            date = all_dates[i]
            path = f'/home/lakeclean/Documents/speciale/target_analysis/'
            path+= f'{all_IDs[i]}/{date[:len("2017-06-04")]}/data/bf_fit_params.txt'
            jd = Time(date).jd #Correcting jd the way Frank likes. Something to with Tess.
            v_bary = all_vbary[i]
            epoch_jds.append(jd)

            #importing the values determined in analyse step
            df = pd.read_csv(path)
            vrad1s = df['epoch_vrad1'].to_numpy() + v_bary
            vrad2s = df['epoch_vrad2'].to_numpy() + v_bary
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

            corr_vrad1s = df['epoch_vrad1'].to_numpy() + v_bary
            corr_vrad2s = df['epoch_vrad2'].to_numpy() + v_bary
            weights = []
            #making corrected vrad:
            for j in range(len(ccds)):
                corr_vrad1s[j] -= ccd_order_offsets[(ccds[j],orders[j])]
                corr_vrad2s[j] -= ccd_order_offsets[(ccds[j],orders[j])]
                weights.append(ccd_order_weights[(ccds[j],orders[j])])
            weights = np.array(weights)         


            #We plot the vrad to determine where the range is good:
            if plotting:
                fig, ax  = plt.subplots()
                ax.scatter(bin_wls,vrad1s,label='primary')
                ax.scatter(bin_wls,vrad2s,label='secondary')
                ax.scatter(bin_wls,corr_vrad1s,label='corrected primary')
                ax.set_xlabel('middle of wavelength range [Å]')
                ax.set_ylabel('Radial velocity [km/s]')
                ax.set_title(f'ID: {ID}, date: {date}')
                ax.legend()
                plt.show()


            #Stating what wavelength range is good
            limit_path = f'/home/lakeclean/Documents/speciale/target_analysis/'
            limit_path += f'{all_IDs[i]}/{date[:len("2017-06-04")]}/data/limits.txt'
            if ask_for_limits:
                
                try:
                    start = int(input('input integer start wl: '))
                    end = int(input('input integer end wl: '))
                    f = open(limit_path, 'w')
                    f.write('start,end\n')
                    f.write(f'{start},{end}')
                    f.close()
                except:
                    print('You inputed something that is not an integer!')
                    start = int(input('input integer start wl: '))
                    end = int(input('input integer end wl: '))
                    f = open(limit_path, 'w')
                    f.write('start,end\n')
                    f.write(f'{start},{end}')
                    f.close()
            else:
                df = pd.read_csv(limit_path)
                start = df['start'].to_numpy()[0]
                end = df['end'].to_numpy()[0]
                print(f'The prechosen limits are: [{start},{end}]')
                
            
                
            #Index inside the specified good range
            idx = np.where( (bin_wls <end) & (bin_wls >start))[0]
            
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
            corr_vrad1s = corr_vrad1s[idx]
            corr_vrad2s = corr_vrad2s[idx]
            weights = weights[idx]

            #normalized weights
            norm_weights = weights / sum(weights)
           
            #Weighted mean
            epoch_rv1s.append(sum(corr_vrad1s*norm_weights))
            epoch_rv2s.append(sum(corr_vrad2s*norm_weights))

            #standard err of weighted mean from unc propagation        
            epoch_rv1_errs.append(np.std(corr_vrad1s)*np.sqrt(sum(norm_weights**2)))
            epoch_rv2_errs.append(np.std(corr_vrad2s)*np.sqrt(sum(norm_weights**2)))
            

        #Exemption for 'KIC-12317678' We here need to find the rv of weak component differently
        if ID == 'KIC12317678':
            distances = []
            for k in [0,20]: #We split the spectrum into two
                path = '/home/lakeclean/Documents/speciale/target_analysis/'
                date_len = len(path + ID)+1
                folder_dates = []
                KIC12317678_dates = []
                for date,IDs in zip(all_dates,all_IDs):
                    if IDs == ID:
                        folder_dates.append(path + ID + f'/{date}')
                        KIC12317678_dates.append(date)
                    
                
                def sorter(x):
                    return Time(x[date_len:]).mjd
                
                folder_dates = sorted(folder_dates,key=sorter)
                
                num_dates = len(folder_dates)
                bfs = np.zeros(shape=(401,num_dates))
                smoothed = np.zeros(shape=(num_dates,401))
                rvs = np.zeros(shape=(401,num_dates))

                # To reveal weak component we take mean of a number of spectra
                order_sum = 20 #the number of orders that is summed
                for i,folder_date in enumerate(folder_dates):
                    for j in range(order_sum):
                        try:
                            df = pd.read_csv(folder_date + f'/data/order_{j+k+20}_broadening_function.txt')
                        except:
                            print(folder_date+' could not be found. If 2024-07-13T00:26:25.672 then its a bad spec')
                            continue
                        df = df.to_numpy()
                        rvs[:,i] = df[:,0]
                        smoothed[i,:] += df[:,2]
                        
                smoothed = smoothed/order_sum
                
                IDlines = open('/home/lakeclean/Documents/speciale/spectra_log_h_readable.txt').read().split('&')
                SB2_IDs, SB2_dates, SB_types, vrad1_guess, vrad2_guess = [], [], [], [], []
                for IDline in IDlines[:-1]:
                    if IDline.split(',')[0][11:].strip(' ') == 'KIC12317678':
                        for line in IDline.split('\n')[2:-1]:
                            line = line.split(',')
                            if line[2].split('/')[0].strip(' ') == 'NaN':
                                continue
                            if line[0].strip(' ') in SB2_dates: #weeding out the duplicates
                                continue
                            if line[0].strip(' ') not in KIC12317678_dates: #only using spectra of same intrument
                                continue
                            SB2_IDs.append(IDline.split(',')[0][11:].strip(' '))
                            SB2_dates.append(line[0].strip(' '))
                            SB_types.append(int(line[1].strip(' ')))
                            if line[1].strip(' ') == '2':
                                vrad1_guess.append(float(line[2].split('/')[0].strip(' ')))
                                vrad2_guess.append(float(line[2].split('/')[1].strip(' ')))
                            else:
                                vrad1_guess.append(0)
                                vrad2_guess.append(0)
                distance = []
                #Fitting each of the mean bfs, to find the distance between peaks
                for i,SB_type in enumerate(SB_types):
                    if SB_type == 1:
                        
                        fit, model, bfgs = shazam.rotbf_fit(rvs[:,i],smoothed[i,:], 30,60000,1, 5,False)
                        distance.append(0)
                        
                    if SB_type == 2:

                        fit, model, bfgs = shazam.rotbf2_fit(rvs[:,i],smoothed[i,:], 30,60000,1,
                                                             5,5,vrad1_guess[i],vrad2_guess[i],0.05,0.05,False, True)
                        vrad1,vrad2 = fit.params['vrad1'].value, fit.params['vrad2'].value
                        ampl1, ampl2 = fit.params['ampl1'].value,fit.params['ampl2'].value

                        if ampl1 > ampl2:
                            if vrad1 < vrad2:
                                distance.append(abs(vrad1 - vrad2))
                            if vrad2 < vrad1:
                                distance.append(-abs(vrad1-vrad2))
                        if ampl2 > ampl1:
                            if vrad2 < vrad1:
                                distance.append(abs(vrad1 - vrad2))
                            if vrad1 < vrad2:
                                distance.append(-abs(vrad1-vrad2))
                
                    
                distances.append(distance)
            
            dist_mean = []
            dist_err = []
            for i,j in zip (np.array(distances)[0,:],np.array(distances)[1,:]):
                dist_mean.append(np.mean([i,j]))
                dist_err.append( np.std([i,j])/np.sqrt(2))
                
            for i in range(len(epoch_rv2s)):
                epoch_rv2s[i] = epoch_rv2s[i] + dist_mean[i]
                epoch_rv2_errs[i] = np.sqrt(epoch_rv2_errs[i]**2 + dist_err[i]**2)
        #######################  End of exemption for KIC-123... ########################

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
        rv_path = f'/home/lakeclean/Documents/speciale/rv_data/{data_type}_{ID}.txt'
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

        
        

        #The mean vsini over each epoch and the error:
        '''
        norm_vsini1_weights = epoch_vsini1_errs / sum(epoch_vsini1_errs)
        norm_vsini2_weights = epoch_vsini2_errs / sum(epoch_vsini2_errs)
        weighted_vsini1_mean = sum(norm_vsini1_weights*epoch_vsini1s)
        weighted_vsini2_mean = sum(norm_vsini2_weights*epoch_vsini2s)
        e_weighted_vsini1_mean = np.std(epoch_vsini1s)*np.sqrt(sum(norm_vsini1_weights**2))
        e_weighted_vsini2_mean = np.std(epoch_vsini2s)*np.sqrt(sum(norm_vsini2_weights**2))
        print(weighted_vsini1_mean,weighted_vsini2_mean,e_weighted_vsini1_mean,e_weighted_vsini2_mean)
        mt.add_value(weighted_vsini1_mean,'vsini1',ID)
        mt.add_value(weighted_vsini2_mean,'vsini2',ID)
        mt.add_value(e_weighted_vsini1_mean,'e_vsini1',ID)
        mt.add_value(e_weighted_vsini2_mean,'e_vsini2',ID)
        '''
    
        
        
data_types = ['NOT', 'NOT_old_HIRES', 'NOT_old_LOWRES', 'TNG']#, 'KECK']
log_paths = ['/home/lakeclean/Documents/speciale/NOT_order_file_log.txt',
             '/home/lakeclean/Documents/speciale/NOT_old_HIRES_order_file_log.txt',
             '/home/lakeclean/Documents/speciale/NOT_old_LOWRES_order_file_log.txt',
             '/home/lakeclean/Documents/speciale/TNG_merged_file_log.txt']

#data_types = ['NOT_old_LOWRES']
#log_paths = ['/home/lakeclean/Documents/speciale/NOT_old_LOWRES_order_file_log.txt']

#data_types = ['NOT']
#log_paths = ['/home/lakeclean/Documents/speciale/NOT_order_file_log.txt']

data_types = ['KECK']
log_paths = ['/home/lakeclean/Documents/speciale/KECK_order_file_log.txt']

for data_type, log_path in zip(data_types, log_paths):
    '''
    #KIC12317678
    if True:
        find_rv_time('KIC12317678',
                     log_path=log_path,
                     data_type=data_type)

    #KIC9693187
    if True:
        find_rv_time('KIC9693187',
                     log_path=log_path,
                     data_type=data_type)
   

    #KIC4914923
    if True:
        find_rv_time('KIC4914923',
                     log_path=log_path,
                     data_type=data_type,
                     limits = [20,60])
    


    #KIC9025370
    if True:
        find_rv_time('KIC9025370',
                     log_path=log_path,
                     data_type=data_type)
    
'''

    #KIC10454113
    if True:
        find_rv_time('KIC10454113',
                     log_path=log_path,
                     data_type=data_type,
                     ask_for_limits=False)
    '''

    
    #KIC4457331
    if True:
        find_rv_time('KIC4457331',
                     log_path=log_path,
                     data_type=data_type)

    #EPIC246696804
    if True:
        find_rv_time('EPIC246696804',
                     log_path=log_path,
                     data_type=data_type)
        
    #EPIC212617037
    if True:
        find_rv_time('EPIC212617037',
                     log_path=log_path,
                     data_type=data_type)
    #EPIC-249570007
    if True:
        find_rv_time('EPIC249570007',
                     log_path=log_path,
                     data_type=data_type)

    #EPIC-230748783
    if True:
        find_rv_time('EPIC230748783',
                     log_path=log_path,
                     data_type=data_type)

    #EPIC-236224056
    if True:
        find_rv_time('EPIC236224056',
                     log_path=log_path,
                     data_type=data_type)

    #KIC4260884
    if True:
        find_rv_time('KIC4260884',
                     log_path=log_path,
                     data_type=data_type)

    #KIC9652971
    if True:
        find_rv_time('KIC9652971',
                     log_path=log_path,
                     data_type=data_type)

    #KIC4260884
    if True:
        find_rv_time('HD208139',
                     log_path=log_path,
                     data_type=data_type)
    '''























    
