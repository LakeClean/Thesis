import matplotlib.pyplot as plt
import numpy as np
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



def plot_rv_time(ID,fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=False, report_fit=False,
                 make_phase_plot=False,make_river_plot=False,scale_river=[-0.0005,0.01,100],
                 make_table=False,print_mass=False,plot_offset=False,exclude_points=5,
                 res1=2,res2=2,res_off1=0,res_off2=0,
                 find_rv=True):
    '''
    Function for plotting the radial velocity plot.
    Parameters:
        - ID             : [str] The ID of the target
        - fit_param1s    : [list of floats] list of fit params, if empty no fit
        - Save_plots     : [Bool] Save the plot if True
        - limits         : [list] first and last order of spectrum section
        - SB_type        : [int] either 1 or 2 for the type of SB type
        - orbital_period : [float] A guess of the orbital period if fit is not done
        - show_plot      : [bool] if true then show the rv plot
        - report_fit     : [bool] if true then fit is reported.
        - make_phase_plot: [bool] if true then make a phase plot. Requires the period to be known
        - make_river_plot: [bool] if true then we make river plot.
        - scale_river    : [list] three elements for scaling the river_plot. [low,high,n_steps]
        - make_table     : [bool] if true then make table of info
        - print_mass     : [bool] if true estimate and print the minimum mass
        - exclude_points : [float] The value from the v0 where points are excluded from fit
        - res1           : [float] The ylim on residual plot res1*np.std(residuals)
        - re_off1        : [float] The offset of the ylim of the residual plot
        - find_rv        : [bool] Find the rv or just import it from file
        

    Returns:
        - resulting_fit  : if fit_params not empty returns array of fit to rv

    Note:
        - fit_params SB1 : [K_guess, e_guess, w_guess,period_guess, v0_guess]
        - fit_params SB2 : [K1_guess,K2_guess, e_guess, w_guess,period_guess, v0_guess]
    '''
    
    if True:
        path = f'/home/lakeclean/Documents/speciale/NOT_order_file_log.txt'
        lines= open(path).read().split('\n')
        all_IDs, all_dates, all_vhelios,files= [], [], [], []
        all_vbary = {}
        for line in lines[1:-1]:
            line = line.split(',')
            if line[5].strip() == 'F4 HiRes':
                if Time(line[2].strip()).jd > 2457506: #weeding out old observations
                    if line[2].strip() not in all_dates: #Skipping duplicates
                        all_IDs.append(line[0].strip())
                        all_dates.append(line[2].strip())
                        all_vhelios.append(float(line[4].strip()))
                        files.append(line[1].strip())
                        all_vbary[line[2].strip()] = float(line[-2].strip())/1000 #correcting from m/s to km/s
            
        def fit_line(x,y):
            fit = fitting.LinearLSQFitter()
            line_init = models.Linear1D()
            fitted_line = fit(line_init, x, y)
            slope = fitted_line.slope.value
            intercept = fitted_line.intercept.value
            new_x = np.linspace(min(x),max(x),10)
            return x,fitted_line(x),slope

        rvs = np.zeros(shape= (len(files),4))
        jds = np.zeros(len(files))
        offset1s = np.zeros(shape= (len(files),91))
        offset2s = np.zeros(shape= (len(files),91))
        offsets = np.zeros(shape= (len(files),91))
        vrad1s = np.zeros(shape= (len(files),91))
        vrad2s= np.zeros(shape= (len(files),91))
        ampl1s= np.zeros(shape= (len(files),91))
        ampl2s= np.zeros(shape= (len(files),91))
        vsini1s= np.zeros(shape= (len(files),91))
        vsini2s= np.zeros(shape= (len(files),91))
        gwidths= np.zeros(shape= (len(files),91))
        limbds= np.zeros(shape= (len(files),91))
        consts= np.zeros(shape= (len(files),91))
        

        flux_levels = []
        

        #Going through each file
        for i,date in enumerate(all_dates):
            path = f'/home/lakeclean/Documents/speciale/target_analysis/'+ f'{all_IDs[i]}/{date}/data/bf_fit_params.txt'
            date = all_dates[i]
            jds[i] = Time(date).jd - 2457000 #Correcting jd the way Frank likes. Something to with Tess.
            v_helio = all_vhelios[i]
            v_bary = all_vbary[date]

            try:
                lines = open(path).read().split('\n')
            except:
                print(f'{path} could not be found')
                continue
                
            #Going through each order
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
                    ampl1s[i,j] = ampl1
                    ampl2s[i,j] = ampl2
                    vrad1s[i,j] =vrad1 + v_bary
                    vrad2s[i,j] = vrad2 + v_bary
                    vsini1s[i,j] = vsini1
                    vsini2s[i,j] = vsini2
                    gwidths[i,j] =gwidth
                    limbds[i,j] =limbd
                    consts[i,j] =const
                else:
                    ampl1s[i,j] = ampl2
                    ampl2s[i,j] = ampl1
                    vrad1s[i,j] =vrad2 + v_bary
                    vrad2s[i,j] = vrad1 + v_bary
                    vsini1s[i,j] = vsini2
                    vsini2s[i,j] = vsini1
                    gwidths[i,j] =gwidth
                    limbds[i,j] =limbd
                    consts[i,j] =const


            
            offset1s[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
            offset2s[i,:] = np.median(vrad2s[i][limits[0]:limits[1]]) - vrad2s[i]

            spectral_type = 1
            for k, j in zip(vrad1s[i],vrad2s[i]):
                if k==j:
                    pass
                else:
                    #print(all_IDs[i])
                    spectral_type=2
                    break
                
            if spectral_type == 1: #Checking if all rads are the same for both components i.e. SB1.
                offsets[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
            if spectral_type == 2:
                offsets[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
                offsets = np.append(offsets,[np.median(vrad2s[i][limits[0]:limits[1]]) - vrad2s[i]],axis=0)
                #offsets[i,:] += np.median(vrad2s[i][limits[0]:limits[1]]) - vrad2s[i]
                #offsets[i,:] = offsets[i,:]/2
                
            #Finding the offset from the median of the best region:
            
            '''
            if all_IDs in ['KIC-9693187','KIC-9025370','KIC9652971']
                offsets[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
                offsets[i,:] += np.median(vrad2s[i][limits[0]:limits[1]]) - vrad2s[i]
                offsets[i,:] = offsets[i,:]/2
            else:
                offsets[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
            '''
                
        #The offset for each order based on every file  
        #offset1s_median = np.median(offset1s,axis=0)
        #offset2s_median = np.median(offset2s,axis=0)
        offsets_median = np.median(offsets,axis=0)

        if False:#plot_offset:
            fig,ax = plt.subplots()
            for i in range(len(vrad1s[:,0])):
                ax.scatter(range(len(offset2s[i,:])),offset2s[i,:])
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
        epoch_vhelio = []
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
                #print(ID,all_ID)
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
            epoch_vbary.append(all_vbary[all_dates[i]])
            epoch_vhelio.append(all_vhelios[i])

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
            for k in [0,20]:
                
                path = '/home/lakeclean/Documents/speciale/target_analysis/'
                date_len = len(path + ID)+1
                folder_dates = glob.glob(path + ID + '/*')
                
                def sorter(x):
                    return Time(x[date_len:]).mjd
                folder_dates = sorted(folder_dates,key=sorter)
                num_dates = len(folder_dates)
                bfs = np.zeros(shape=(401,num_dates))
                smoothed = np.zeros(shape=(num_dates,401))
                rvs = np.zeros(shape=(401,num_dates))

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

                #Checking out the summed bfs:
                '''
                fig,ax = plt.subplots()
                for i in range(num_dates):
                    offset = Time(folder_dates[i][date_len:]).mjd*0.01
                    ax.plot(rvs[:,i],smoothed[i,:] + offset)
                plt.show()
                '''
                    
                
                IDlines = open('/home/lakeclean/Documents/speciale/spectra_log_h_readable.txt').read().split('&')
                SB2_IDs, SB2_dates, SB_types, vrad1_guess, vrad2_guess = [], [], [], [], []
                for IDline in IDlines[:-1]:
                    if IDline.split(',')[0][11:].strip(' ') == 'KIC12317678':
                        #print(IDline.split(',')[0][11:].strip(' '))
                        for line in IDline.split('\n')[2:-1]:
                            line = line.split(',')
                            if line[2].split('/')[0].strip(' ') == 'NaN':
                                continue
                            if line[0].strip(' ') in SB2_dates:
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
                for i,SB_type in enumerate(SB_types):
                    if SB_type == 1:
                        
                        fit, model, bfgs = shazam.rotbf_fit(rvs[:,i],smoothed[i,:], 30,60000,1, 5,False)
                        distance.append(0)
                        
                    if SB_type == 2:

                        fit, model, bfgs = shazam.rotbf2_fit(rvs[:,i],smoothed[i,:], 30,60000,1,
                                                             5,5,vrad1_guess[i],vrad2_guess[i],0.05,0.05,False, True)
                        vrad1,vrad2 = fit.params['vrad1'].value, fit.params['vrad2'].value
                        ampl1, ampl2 = fit.params['ampl1'].value,fit.params['ampl2'].value

                        #vel = np.linspace(-200,200,1000)
                        #peak1 = max(shazam.rotbf_func(vel,ampl1,vrad1,vsini1,gwidth,const,limbd))
                        #peak2 = max(shazam.rotbf_func(vel,ampl2,vrad2,vsini2,gwidth,const,limbd))
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
                SB_type=2
            
            dist_mean = []
            dist_err = []
            for i,j in zip (np.array(distances)[0,:],np.array(distances)[1,:]):
                dist_mean.append(np.mean([i,j]))
                dist_err.append( np.std([i,j])/np.sqrt(2))


            for i in range(len(epoch_rv2s)):
                epoch_rv2s[i] = epoch_rv2s[i] + dist_mean[i]
                epoch_rv2_errs[i] = np.sqrt(epoch_rv2_errs[i]**2 + dist_err[i]**2)
        # End of exemption for KIC-123...
        

        epoch_rv1s = np.array(epoch_rv1s)
        epoch_rv2s = np.array(epoch_rv2s)
        epoch_rv1_errs = np.array(epoch_rv1_errs)
        epoch_rv2_errs = np.array(epoch_rv2_errs)
        epoch_jds = np.array(epoch_jds)
        epoch_dates = np.array(epoch_dates)
        epoch_vbary = np.array(epoch_vbary)
        epoch_vhelio = np.array(epoch_vhelio)
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
        rv_path = f'/home/lakeclean/Documents/speciale/rv_data/NOT_{ID}.txt'
        rv_names = ['jd','date','rv1','rv2','e_rv1','e_rv2','vbary','vhelio',
                    'vsini1','vsini2','e_vsini1','e_vsini2','ampl1','ampl2', 'const', 'gwidth', 'limbd']
        
        rv_dat = [epoch_jds,epoch_dates,epoch_rv1s,epoch_rv2s,
                  epoch_rv1_errs,epoch_rv2_errs,epoch_vbary,epoch_vhelio,
                  epoch_vsini1s, epoch_vsini2s,epoch_vsini1_errs, epoch_vsini2_errs,
                  epoch_ampl1s, epoch_ampl2s, epoch_consts, epoch_gwidths, epoch_limbds]
        rv_dict = {}
        for i in range(len(rv_names)):
            rv_dict[rv_names[i]] = rv_dat[i]
        df = pd.DataFrame(rv_dict)
        df.to_csv(rv_path,index=False)

        #The mean vsini over each epoch and the error:
        norm_vsini1_weights = epoch_vsini1_errs / sum(epoch_vsini1_errs)
        norm_vsini2_weights = epoch_vsini2_errs / sum(epoch_vsini2_errs)
        weighted_vsini1_mean = sum(norm_vsini1_weights*epoch_vsini1s)
        weighted_vsini2_mean = sum(norm_vsini2_weights*epoch_vsini2s)
        e_weighted_vsini1_mean = np.std(epoch_vsini1s)*np.sqrt(sum(norm_vsini1_weights**2))
        e_weighted_vsini2_mean = np.std(epoch_vsini2s)*np.sqrt(sum(norm_vsini2_weights**2))
        print(epoch_vsini2s)
        print(weighted_vsini1_mean,weighted_vsini2_mean,e_weighted_vsini1_mean,e_weighted_vsini2_mean)
        mt.add_value(weighted_vsini1_mean,'vsini1',ID)
        mt.add_value(weighted_vsini2_mean,'vsini2',ID)
        mt.add_value(e_weighted_vsini1_mean,'e_vsini1',ID)
        mt.add_value(e_weighted_vsini2_mean,'e_vsini2',ID)
    
        
        



#KIC-12317678
if True:
    plot_rv_time('KIC12317678',limits=[20,60],SB_type=2,find_rv=True)



#KIC-9693187
if True:
    plot_rv_time('KIC9693187',find_rv=True)

#KIC-4914923
if True:
    plot_rv_time('KIC4914923',SB_type=1)


#KIC-9025370
if True:
    plot_rv_time('KIC9025370', SB_type=2) 


#KIC-10454113
if True:
    plot_rv_time('KIC10454113', SB_type=1)



#KIC4457331
if True:
    plot_rv_time('KIC4457331', SB_type=1)

#EPIC-246696804
if True:
    plot_rv_time('EPIC246696804',SB_type=1)
    
#EPIC-212617037
if True:
    plot_rv_time('EPIC212617037' ,SB_type=1)
    
#EPIC-249570007
if True:
    plot_rv_time('EPIC249570007', SB_type=1)


#EPIC-230748783
if True:
    plot_rv_time('EPIC230748783', SB_type=1)


#EPIC-236224056
if True:
    plot_rv_time('EPIC236224056', SB_type=1)

#KIC4260884
if True:
    plot_rv_time('KIC4260884', SB_type=1)

#KIC9652971
if True:
    plot_rv_time('KIC9652971', SB_type=1)

#HD208139
if True:
    plot_rv_time('HD208139', SB_type=1)





















    
