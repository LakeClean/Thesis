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
def find_rv_time(ID,log_path,data_type, limits=[0,-1],plot_offset=True):
    '''
    Function for plotting the radial velocity plot.
    Parameters:
        - ID             : [str] The ID of the target
        - log_path       : [str] The path to the log of the specific dataset we want to analyse
        - limits         : [list] first and last order of spectrum section
        

    Returns:
        prints resulting rv estimate for each epoch to file

    '''
    data_types = ['NOT', 'NOT_old_HIRES', 'NOT_old_LOWRES', 'TNG', 'KECK']
    if data_type not in data_types:
        print(f'The given data_type is not in {data_types}')
        return 0
    
    if True:
        path = log_path
        df = pd.read_csv(log_path)
        all_IDs = df['ID'].to_numpy()
        all_dates = df['date'].to_numpy()
        files = df['directory'].to_numpy()
        all_vbary = df['v_bary'].to_numpy()/1000
        '''
        lines= open(path).read().split('\n')
        all_IDs, all_dates,files= [], [], []
        all_vbary = {}
        for line in lines[1:-1]:
            line = line.split(',')
            if line[2].strip() not in all_dates: #Skipping duplicates
                    all_IDs.append(line[0].strip())
                    all_dates.append(line[2].strip())
                    #all_vhelios.append(float(line[5].strip()))
                    files.append(line[1].strip())
                    all_vbary[line[2].strip()] = float(line[-2].strip())/1000 #correcting from m/s to km/s
        '''
            
        def fit_line(x,y):
            fit = fitting.LinearLSQFitter()
            line_init = models.Linear1D()
            fitted_line = fit(line_init, x, y)
            slope = fitted_line.slope.value
            intercept = fitted_line.intercept.value
            new_x = np.linspace(min(x),max(x),10)
            return x,fitted_line(x),slope

        #Finding the number of bins:
        lines = open(f'/home/lakeclean/Documents/speciale/target_analysis/'+ f'{all_IDs[0]}/{all_dates[0]}/data/bf_fit_params.txt').read().split('\n')[1:-1]
        n_bins = len(lines)
        print(f'Analyzing target: {ID}')
        print(f'Using the logfile: {log_path}')
        print(f'RV-data will be saved as: "/home/lakeclean/Documents/speciale/rv_data/{data_type}_{ID}.txt" ')
        print(f'Number of bins: {n_bins}')

        rvs = np.zeros(shape= (len(files),4))
        jds = np.zeros(len(files))
        offset1s = np.zeros(shape= (len(files),n_bins))
        offset2s = np.zeros(shape= (len(files),n_bins))
        offsets = np.zeros(shape= (len(files),n_bins))
        
        vrad1s = np.zeros(shape= (len(files),n_bins))
        vrad2s= np.zeros(shape= (len(files),n_bins))
        ampl1s= np.zeros(shape= (len(files),n_bins))
        ampl2s= np.zeros(shape= (len(files),n_bins))
        vsini1s= np.zeros(shape= (len(files),n_bins))
        vsini2s= np.zeros(shape= (len(files),n_bins))
        gwidths= np.zeros(shape= (len(files),n_bins))
        limbds= np.zeros(shape= (len(files),n_bins))
        consts= np.zeros(shape= (len(files),n_bins))
        

        flux_levels = []
        #Going through each file
        for i,date in enumerate(all_dates):
            path = f'/home/lakeclean/Documents/speciale/target_analysis/'+ f'{all_IDs[i]}/{date}/data/bf_fit_params.txt'
            date = all_dates[i]
            jds[i] = Time(date).jd #Correcting jd the way Frank likes. Something to with Tess.
            v_bary = all_vbary[i]

            try:
                lines = open(path).read().split('\n')
            except:
                print(f'{path} could not be found')
                continue
            
                
            #Going through each bin
            for j,line in enumerate(lines[1:-1]):
                line  =line.split(',')
                ampl1 = float(line[2])
                ampl2 = float(line[3])
                vrad1 =float(line[0]) 
                vrad2 = float(line[1])
                vsini1 = float(line[4])
                vsini2 = float(line[5])
                gwidth =float(line[6])
                limbd =float(line[7])
                const =float(line[8])

                vel = np.linspace(-200,200,1000)
                peak1 = max(shazam.rotbf_func(vel,ampl1,vrad1,vsini1,gwidth,const,limbd))
                peak2 = max(shazam.rotbf_func(vel,ampl2,vrad2,vsini2,gwidth,const,limbd))

                #Picking the maximum of the BFs as 1st component
                if peak1 < peak2:
                    ampl1s[i,j] = ampl2
                    ampl2s[i,j] = ampl1
                    vrad1s[i,j] =vrad2 + v_bary
                    vrad2s[i,j] = vrad1 + v_bary
                    vsini1s[i,j] = vsini2
                    vsini2s[i,j] = vsini1
                    gwidths[i,j] =gwidth
                    limbds[i,j] =limbd
                    consts[i,j] =const
                    
                else:
                    ampl1s[i,j] = ampl1
                    ampl2s[i,j] = ampl2
                    vrad1s[i,j] =vrad1 + v_bary
                    vrad2s[i,j] = vrad2 + v_bary
                    vsini1s[i,j] = vsini1
                    vsini2s[i,j] = vsini2
                    gwidths[i,j] =gwidth
                    limbds[i,j] =limbd
                    consts[i,j] =const
                    


            offset1s[i,:] = np.median(vrad1s[i]) - vrad1s[i]
            offset2s[i,:] = np.median(vrad2s[i]) - vrad2s[i]

            spectral_type = 1
            for k, j in zip(vrad1s[i],vrad2s[i]):
                if k==j:
                    pass
                else:
                    spectral_type=2
                    break
                
            if spectral_type == 1: #Checking if all rads are the same for both components i.e. SB1.
                offsets[i,:] = offset1s[i,:]
            if spectral_type == 2:
                offsets[i,:] = offset1s[i,:]
                offsets = np.append(offsets,[offset2s[i,:]],axis=0)
            
                
        #The offset for each order based on every file  
        offsets_median = np.median(offsets,axis=0)
        if plot_offset:
            fig,ax = plt.subplots()
            for i in range(len(vrad1s[:,0])):
                ax.scatter(range(len(offsets[i,:])),offsets[i,:])

            plt.show()
            plt.close()
        
        weight1s = np.std(offsets,axis=0)
        weight2s = np.std(offsets,axis=0) #Relic of before offsets were evaluated on the same basis

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
        
        for i,all_ID in enumerate(all_IDs):
            if not ID == all_ID:
                continue

            #We choose the vsini as the mean of the order vsini's that lie in good range:
            mean_vsini1 = np.mean(vsini1s[i][limits[0]:limits[1]])
            err_vsini1 = np.std(vsini1s[i][limits[0]:limits[1]])/len(vsini1s[i][limits[0]:limits[1]])
            mean_vsini2 = np.mean(vsini2s[i][limits[0]:limits[1]])
            err_vsini2 = np.std(vsini2s[i][limits[0]:limits[1]])/len(vsini2s[i][limits[0]:limits[1]])
            epoch_vsini1s.append(mean_vsini1)
            epoch_vsini2s.append(mean_vsini2)
            epoch_vsini1_errs.append(err_vsini1)
            epoch_vsini2_errs.append(err_vsini2)

            
            epoch_gwidths.append(np.median(gwidths[i][limits[0]:limits[1]]))
            epoch_limbds.append(np.median(limbds[i][limits[0]:limits[1]]))
            epoch_ampl1s.append(np.median(ampl1s[i][limits[0]:limits[1]]))
            epoch_ampl2s.append(np.median(ampl2s[i][limits[0]:limits[1]]))
            epoch_consts.append(np.median(consts[i][limits[0]:limits[1]]))
            
            
            epoch_dates.append(all_dates[i])
            epoch_vbary.append(all_vbary[i])

            #Correcting vrads for NOT wavelength issue
            corr_vrad1s = vrad1s[i]+offsets_median
            corr_vrad2s = vrad2s[i]+offsets_median

            #Selecting the best region of orders
            corr_vrad1s = corr_vrad1s[limits[0]:limits[1]]
            corr_vrad2s = corr_vrad2s[limits[0]:limits[1]]
            corr_weight1s = weight1s[limits[0]:limits[1]]
            corr_weight2s = weight2s[limits[0]:limits[1]]

            #picking out outliers
            good_vrad1s = []
            good_vrad2s = []
            good_weight1s = []
            good_weight2s = []
            for y1,y2,w1,w2 in zip(corr_vrad1s,corr_vrad2s,corr_weight1s,corr_weight2s):
                
                outlier_limit = 5
                temp_median1 = np.median(corr_vrad1s)
                temp_median2 = np.median(corr_vrad2s)

                if abs(y1 - temp_median1) < outlier_limit:
                    good_vrad1s.append(y1)
                    good_weight1s.append(w1)

                if abs(y2 - temp_median2) < outlier_limit:
                    good_vrad2s.append(y2)
                    good_weight2s.append(w2)

            #normalized weights
            epoch_norm_weight1s = np.array(good_weight1s) / sum(np.array(good_weight1s))
            epoch_norm_weight2s = np.array(good_weight2s) / sum(np.array(good_weight2s))

            #Weighted mean
            epoch_rv1s.append(sum(np.array(good_vrad1s)*epoch_norm_weight1s))
            epoch_rv2s.append(sum(np.array(good_vrad2s)*epoch_norm_weight2s))

            #standard err of weighted mean from unc propagation        
            epoch_rv1_errs.append(np.std(good_vrad1s)*np.sqrt(sum(epoch_norm_weight1s**2)))#append(np.std(good_vrad1s)/np.sqrt(len(good_vrad1s)))
            epoch_rv2_errs.append(np.std(good_vrad2s)*np.sqrt(sum(epoch_norm_weight2s**2)))#append(np.std(good_vrad2s)/np.sqrt(len(good_vrad2s)))
            epoch_jds.append(jds[i])

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
    
        
        
#data_types = ['NOT', 'NOT_old_HIRES', 'NOT_old_LOWRES', 'TNG']#, 'KECK']
#log_paths = ['/home/lakeclean/Documents/speciale/NOT_order_file_log.txt',
#             '/home/lakeclean/Documents/speciale/NOT_old_HIRES_order_file_log.txt',
#             '/home/lakeclean/Documents/speciale/NOT_old_LOWRES_order_file_log.txt',
#             '/home/lakeclean/Documents/speciale/TNG_merged_file_log.txt']

#data_types = ['NOT_old_LOWRES']
#log_paths = ['/home/lakeclean/Documents/speciale/NOT_old_LOWRES_order_file_log.txt']

data_types = ['NOT']
log_paths = ['/home/lakeclean/Documents/speciale/NOT_order_file_log.txt']

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
    


    #KIC10454113
    if True:
        find_rv_time('KIC10454113',
                     log_path=log_path,
                     data_type=data_type)

    
    #KIC4457331
    if True:
        find_rv_time('KIC4457331',
                     log_path=log_path,
                     data_type=data_type)
    '''

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
    if False:
        find_rv_time('KIC4260884',
                     log_path=log_path,
                     data_type=data_type)

    #KIC9652971
    if False:
        find_rv_time('KIC9652971',
                     log_path=log_path,
                     data_type=data_type)

    #KIC4260884
    if False:
        find_rv_time('HD208139',
                     log_path=log_path,
                     data_type=data_type)
























    
