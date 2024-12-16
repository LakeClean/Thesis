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

#KIC10454113:
V_litt = [-22.69,-20.683]
e_V_litt = [0.39,0]
V_litt_time = [175.32222325215116,-568.0389999998733]




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

    '''
    if find_rv:
        path = f'/home/lakeclean/Documents/speciale/old/order_file_log.txt'
        lines= open(path).read().split('\n')
        all_IDs, all_dates, all_vhelios,files= [], [], [], []
        all_vbary = {}
        for line in lines[:-1]:
            line = line.split(',')
            if line[1].strip() == 'science':
                if line[3].strip() not in all_dates: #Skipping duplicates
                    all_IDs.append(line[0].strip())
                    all_dates.append(line[3].strip())
                    all_vhelios.append(float(line[5].strip()))
                    files.append(line[2].strip())
                    all_vbary[line[3].strip()] = float(line[-2].strip())/1000 #correcting from m/s to km/s
            
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
                
            
                
        #The offset for each order based on every file  
        #offset1s_median = np.median(offset1s,axis=0)
        #offset2s_median = np.median(offset2s,axis=0)
        offsets_median = np.median(offsets,axis=0)

        if plot_offset:
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
                continue
            print(vsini1s)
            print(vsini2s)

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
        print(weighted_vsini1_mean,weighted_vsini2_mean,e_weighted_vsini1_mean,e_weighted_vsini2_mean)
        mt.add_value(weighted_vsini1_mean,'vsini1',ID)
        mt.add_value(weighted_vsini2_mean,'vsini2',ID)
        mt.add_value(e_weighted_vsini1_mean,'e_vsini1',ID)
        mt.add_value(e_weighted_vsini2_mean,'e_vsini2',ID)
        
        

    else:
        df = pd.read_csv(f'/home/lakeclean/Documents/speciale/rv_data/NOT_{ID}.txt')
        rv_names = ['jd','date','rv1','rv2','e_rv1','e_rv2','vbary','vhelio']
        epoch_rv1s = df['rv1'].to_numpy()
        epoch_rv2s = df['rv2'].to_numpy()
        epoch_rv1_errs = df['e_rv1'].to_numpy()
        epoch_rv2_errs = df['e_rv2'].to_numpy()
        epoch_jds = df['jd'].to_numpy()
        epoch_dates = df['date'].to_numpy()
        epoch_vbary = df['vbary'].to_numpy()
        epoch_vhelio = df['vhelio'].to_numpy()
    '''


    df = pd.read_csv(f'/home/lakeclean/Documents/speciale/rv_data/NOT_{ID}.txt')
    rv_names = ['jd','date','rv1','rv2','e_rv1','e_rv2','vbary','vhelio']
    epoch_rv1s = df['rv1'].to_numpy()
    epoch_rv2s = df['rv2'].to_numpy()
    epoch_rv1_errs = df['e_rv1'].to_numpy()
    epoch_rv2_errs = df['e_rv2'].to_numpy()
    epoch_jds = df['jd'].to_numpy()
    epoch_dates = df['date'].to_numpy()
    epoch_vbary = df['vbary'].to_numpy()
    epoch_vhelio = df['vhelio'].to_numpy()

    df = pd.read_csv(f'/home/lakeclean/Documents/speciale/rv_data/TNG_{ID}.txt')
    TNG_rv1s = df['rv1'].to_numpy()
    TNG_rv2s = df['rv2'].to_numpy()
    TNG_rv1_errs = df['e_rv1'].to_numpy()
    TNG_rv2_errs = df['e_rv2'].to_numpy()
    TNG_jds = df['jd'].to_numpy()
    TNG_dates = df['date'].to_numpy()
    TNG_vbary = df['vbary'].to_numpy()
    
    


    #Plotting the radial velocity:
    if SB_type == 1:
        fig,ax = plt.subplots(2,1,sharex = True, height_ratios=[3,1])
        fig.subplots_adjust(hspace=0)
        ax[1].set_xlabel(f'JD - 2457000 [days]')
    if SB_type == 2:
        fig,ax = plt.subplots(3,1,sharex = True, height_ratios=[3,1,1])
        fig.subplots_adjust(hspace=0)
        ax[2].set_xlabel(f'JD - 2457000 [days]')
    ax[0].set_title(f'ID: {ID}')
    ax[0].set_ylabel('radial velocity [km/s]')
    
    if (len(fit_params)> 0):
        #finding index of those points that lie too close (5km/s) to the system velocity
        v0_guess = fit_params[-1]
        included1 = np.where(abs(np.array(epoch_rv1s) - v0_guess) >= exclude_points)[0]
        excluded1 = np.where(abs(np.array(epoch_rv1s) - v0_guess) < exclude_points)[0]
        included2 = np.where(abs(np.array(epoch_rv2s) - v0_guess) >= exclude_points)[0]
        excluded2 = np.where(abs(np.array(epoch_rv2s) - v0_guess) < exclude_points)[0]

        ax[0].errorbar(epoch_jds[included1], epoch_rv1s[included1], epoch_rv1_errs[included1],
                   fmt='o',capsize=2,color='r')
        
        ax[0].errorbar(epoch_jds[excluded1], epoch_rv1s[excluded1], epoch_rv1_errs[excluded1],
                   fmt='x',capsize=2,color='grey')
            
        ax[0].errorbar(epoch_jds[included2], epoch_rv2s[included2], epoch_rv2_errs[included2],
                   fmt='o',capsize=2,color='b')
        
        ax[0].errorbar(epoch_jds[excluded2], epoch_rv2s[excluded2], epoch_rv1_errs[excluded2],
                   fmt='x',capsize=2,color='grey')
        
        ax[0].errorbar(TNG_jds, TNG_rv1s, TNG_rv1_errs,
                   fmt='o',capsize=2,color='green')

        ax[0].errorbar(TNG_jds, TNG_rv2s, TNG_rv2_errs,
                   fmt='o',capsize=2,color='green')
            

    else:
        ax[0].errorbar(epoch_jds, epoch_rv1s, epoch_rv1_errs,
                   fmt='o',capsize=2,color='r')

        ax[0].errorbar(epoch_jds, epoch_rv2s, epoch_rv2_errs,
                   fmt='o',capsize=2,color='b')

        ax[0].errorbar(TNG_jds, TNG_rv1s, TNG_rv1_errs,
                   fmt='o',capsize=2,color='green')

        ax[0].errorbar(TNG_jds, TNG_rv2s, TNG_rv2_errs,
                   fmt='o',capsize=2,color='green')

        ax[0].errorbar(V_litt_time, V_litt, e_V_litt,
                   fmt='o',capsize=2,color='purple')

        



    
    print(f'SB Type: ', SB_type)
    if (len(fit_params)> 0) and (SB_type == 1):
        ################### SB1 #######################

        #Fitting
        K_guess, e_guess, w_guess,period_guess, v0_guess = fit_params
                
        fit = sb.fit_radvel_SB1(epoch_jds[included1],epoch_rv1s[included1],k=K_guess,e=e_guess,
                                w=w_guess,p=period_guess,v0=v0_guess)

        k = fit.params['k1'].value
        e = fit.params['e'].value
        w = fit.params['w'].value
        p = fit.params['p'].value
        t0 = fit.params['t0'].value
        v0 = fit.params['v0_1'].value
        covar = np.diag(fit.covar)
        e_k = np.sqrt(covar[0])
        e_e = np.sqrt(covar[1])
        e_w = np.sqrt(covar[2])
        e_p = np.sqrt(covar[3])
        e_t0 = np.sqrt(covar[4])
        e_v0 = np.sqrt(covar[5])

        if make_table:
            values_to_add = [k,e,w,p,v0,e_k,e_e,e_w,e_p,e_v0]
            names_to_add = ['k1','e','w','p','v01','e_k1','e_e','e_w','e_p','e_v01']
            for v,n in zip(values_to_add,names_to_add):
                mt.add_value(v,n,ID)

        if report_fit:
            print(fit.params)

        #Plot of fit
        proxy_time = np.linspace(min(epoch_jds),max(epoch_jds),1000)
        fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
        ax[0].plot(proxy_time,fit_rvs,label='fit')
        ax[0].plot([min(epoch_jds),max(epoch_jds)],[v0,v0],
                   ls='--',color='black',alpha=0.4,label=f'v0={np.round(v0,2)}km/s')
        ax[0].legend()


        #Residuals:
        residual = epoch_rv1s -sb.radial_velocity(epoch_jds,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
        ax[1].set_ylim(-res1*np.std(residual[included1]) + res_off1,
                       res_off1 + res1*np.std(residual[included1]))

        ax[1].errorbar(epoch_jds[included1],residual[included1],epoch_rv1_errs[included1],
               fmt='o',capsize=2,color='r')

        ax[1].errorbar(epoch_jds[excluded1],residual[excluded1],epoch_rv1_errs[excluded1],
               fmt='x',capsize=2,color='grey')
        
        ax[1].plot([min(epoch_jds),max(epoch_jds)],[0,0]
                   ,ls='--',color='black',alpha=0.4)
        
        RMS1 = np.sqrt(np.mean(residual[included1]**2))
        ax[1].plot([min(epoch_jds),max(epoch_jds)],[RMS1,RMS1],
                    ls='-.',color='black',alpha=0.4,
                    label=f'RMS: {np.round(RMS1,2)}')
        ax[1].legend()
        
        

    if (len(fit_params)> 0) and (SB_type == 2):
        ##################### SB2 ######################

        #Fitting
        K1_guess, K2_guess, e_guess, w_guess, period_guess, v0_guess = fit_params

        rvs = [epoch_rv1s[included1],epoch_rv2s[included2],epoch_rv1_errs[included1],epoch_rv2_errs[included2]]
        jds = [epoch_jds[included1],epoch_jds[included2]]
        fit = sb.fit_radvel_SB2(jds,rvs,k=[K1_guess,K2_guess],e=e_guess,
                                w=w_guess,p=period_guess,v0=[v0_guess,v0_guess])
        
        k1 = fit.params['k1'].value
        k2 = fit.params['k2'].value
        e = fit.params['e'].value
        w = fit.params['w'].value
        p = fit.params['p'].value
        t0 = fit.params['t0'].value
        v01 = fit.params['v0_1'].value
        v02 = fit.params['v0_2'].value
        covar = np.diag(fit.covar)
        e_k1 = np.sqrt(covar[0])
        e_k2 = np.sqrt(covar[1])
        e_e = np.sqrt(covar[2])
        e_w = np.sqrt(covar[3])
        e_p = np.sqrt(covar[4])
        e_t0 = np.sqrt(covar[5])
        e_v01 = np.sqrt(covar[6])
        e_v02 = np.sqrt(covar[7])

        if make_table:
            values_to_add = [k1,k2,e,w,p,v01,v02,e_k1,e_k2,e_e,e_w,e_p,e_v01,e_v02,t0]
            names_to_add = ['k1','k2','e','w','p','v01','v02','e_k1','e_k2','e_e','e_w','e_p','e_v01','e_v02','t0']
            for v,n in zip(values_to_add,names_to_add):
                mt.add_value(v,n,ID)
        
        

        if report_fit: print(fit.params)
            
        proxy_time = np.linspace(min(epoch_jds),max(epoch_jds),1000)

        fit_rv1s = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v01)
        ax[0].plot(proxy_time,fit_rv1s,label='fit')

        fit_rv2s = sb.radial_velocity(proxy_time,k=-k2,e=e,w=w,p=p,t0=t0,v0=v02)
        ax[0].plot(proxy_time,fit_rv2s,label='fit')
        
        ax[0].plot([min(epoch_jds),max(epoch_jds)],[v01,v01],
                   ls='--',color='black',alpha=0.4,label=f'v01={np.round(v01,2)}km/s')
        ax[0].plot([min(epoch_jds),max(epoch_jds)],[v02,v02],
                   ls='-.',color='black',alpha=0.4,label=f'v02={np.round(v02,2)}km/s')
        ax[0].legend()
        

        #Residuals:
        residual1 = epoch_rv1s- sb.radial_velocity(epoch_jds,k=k1,e=e,w=w,p=p,t0=t0,v0=v01)
        ax[1].errorbar(epoch_jds[included1],residual1[included1],epoch_rv1_errs[included1],
               fmt='o',capsize=2,color='r')
        ax[1].errorbar(epoch_jds[excluded1],residual1[excluded1],epoch_rv1_errs[excluded1],
               fmt='x',capsize=2,color='grey')
        ax[1].set_ylim(-res1*np.std(residual1[included1]) + res_off1,
                       res_off1 + res1*np.std(residual1[included1]))
        ax[1].plot([min(epoch_jds),max(epoch_jds)],[0,0]
                   ,ls='--',color='black',alpha=0.4)
        
        RMS1 = np.sqrt(np.mean(residual1[included1]**2))
        ax[1].plot([min(epoch_jds),max(epoch_jds)],[RMS1,RMS1],
                   ls='-.',color='black',alpha=0.4,
                   label=f'RMS: {np.round(RMS1,2)}')
        
        residual2 = epoch_rv2s-sb.radial_velocity(epoch_jds,k=-k2,e=e,w=w,p=p,t0=t0,v0=v02)
        ax[2].errorbar(epoch_jds[included2],residual2[included2],epoch_rv2_errs[included2],
               fmt='o',capsize=2,color='b')
        ax[2].errorbar(epoch_jds[excluded2],residual2[excluded2],epoch_rv2_errs[excluded2],
               fmt='x',capsize=2,color='grey')
        ax[2].set_ylim(-res2*np.std(residual2[included2]) + res_off2,
                       res_off2 + res2*np.std(residual2[included2]))
        
        
        ax[2].plot([min(epoch_jds),max(epoch_jds)],[0,0]
                   ,ls='--',color='black',alpha=0.4)

        RMS2 = np.sqrt(np.mean(residual2[included2]**2))
        ax[2].plot([min(epoch_jds),max(epoch_jds)],[RMS2,RMS2],
                   ls='-.',color='black',alpha=0.4,
                   label=f'RMS: {np.round(RMS2,2)}')
        ax[1].legend()
        ax[2].legend()
        

        

    plot_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/'
    if save_plot: fig.savefig(plot_path+f"rv_time_{ID}.pdf",
                                   dpi='figure', format='pdf')
                  
    if show_plot: plt.show()
    plt.close()

    ############################## Minimum mass estimate: ##############################

    def minimum_mass_func(k,e,p,e_k,e_e,e_p):
        
        sun_mass = 1.988 * 10**30 #kg
        G  =6.674*10**(-11) #N m^2 / kg ^2
        min_mass_func = ((k*1000)**3 * p*24*60*60 * (1-e**2)**(3/2) / (2*np.pi*G))/sun_mass

        varsAndVals = {'p':[p,e_p],'k':[k,e_k],'e':[e,e_e]}
        min_mass_func_unc = f'((k*1000)**3 * p*24*60*60 * (1-e**2)**(3/2) / (2*{np.pi}*{G}))/{sun_mass}'
        e_min_mass_func = ophobning(min_mass_func_unc,varsAndVals,False)
        
        return min_mass_func, e_min_mass_func

    def absolute_mass_func(k,e,p,A,B,F,G,e_k,e_e,e_p,e_A,e_B,e_F,e_G):
        
        min_mass_func, e_min_mass_func = minimum_mass_func(k,e,p,e_k,e_e,e_p)
        a,Omega,w,i, e_a,e_Omega,e_w,e_i = find_Campbell(A,B,F,G,e_A,e_B,e_F,e_G)
        abs_mass_func = min_mass_func/(np.sin(i)**3)
        print('incliniation', e_i)
        print(e_min_mass_func)
        
        varsAndVals = {'min_mass_func':[min_mass_func,e_min_mass_func],'i':[i,e_i]}
        abs_mass_func_unc = 'min_mass_func/(sin(i)**3)'
        e_abs_mass_func = ophobning(abs_mass_func_unc,varsAndVals,False)

        if make_table:
                names_to_add = ['G_a', 'G_Omega', 'G_w', 'G_i',
                                 'e_G_a', 'e_G_Omega', 'e_G_w', 'e_G_i']
                values_to_add = [a,Omega,w,i, e_a,e_Omega,e_w,e_i]
                for v,n in zip(values_to_add,names_to_add):
                    mt.add_value(v,n,ID)
        
        return abs_mass_func, e_abs_mass_func
        
        
    def minimum_mass(k1,k2,e,p,e_k1,e_k2,e_e,e_p):
        sun_mass = 1.988 * 10**30 #kg
        G  =6.674*10**(-11) #N m^2 / kg ^2
        #min_mass1 = (1+k1/k2)**2 * p*24*60*60 * (k2*1000)**3 * (1-e**2)**(3/2) / (2*np.pi*G)
        #min_mass2 = (1+k2/k1)**2 * p*24*60*60 * (k1*1000)**3 * (1-e**2)**(3/2) / (2*np.pi*G)
        min_mass1 = ((1+k1/k2)**(-1) * p*24*60*60 * (k2*1000 + k1*1000)**3 * (1-e**2)**(3/2) / (2*np.pi*G) )/sun_mass
        min_mass2 = ((1+k2/k1)**(-1) * p*24*60*60 * (k1*1000+k2*1000)**3 * (1-e**2)**(3/2) / (2*np.pi*G)  )/sun_mass

        varsAndVals = {'p':[p,e_p],'k1':[k1,e_k1],'k2':[k2,e_k2],'e':[e,e_e]}
        min_mass1_unc = f'( (1+k1/k2)**(-1) * p*24*60*60 * (k2*1000 + k1*1000)**3 * (1-e**2)**(3/2) / (2*{np.pi}*{G}) )/{sun_mass}'
        min_mass2_unc = f'( (1+k2/k1)**(-1) * p*24*60*60 * (k1*1000 + k1*1000)**3 * (1-e**2)**(3/2) / (2*{np.pi}*{G}) )/{sun_mass}'
        e_min_mass1 = ophobning(min_mass1_unc,varsAndVals,False)
        e_min_mass2 = ophobning(min_mass2_unc,varsAndVals,False)

        return min_mass1, min_mass2,e_min_mass1,e_min_mass2
    
    
    def absolute_mass(k1,k2,e,p, A,B,F,G,e_k1,e_k2,e_e,e_p,e_A,e_B,e_F,e_G):
        min_mass1, min_mass2,e_min_mass1,e_min_mass2 = minimum_mass(k1,k2,e,p,e_k1,e_k2,e_e,e_p)
        a,Omega,w,i, e_a,e_Omega,e_w,e_i = find_Campbell(A,B,F,G,e_A,e_B,e_F,e_G)

        abs_mass1 = min_mass1 / (np.sin(i)**3)
        abs_mass2 = min_mass2 / (np.sin(i)**3)

        varsAndVals = {'min_mass1':[min_mass1,e_min_mass1],'min_mass2':[min_mass2,e_min_mass2],'i':[i,e_i]}
        abs_mass1_unc = 'min_mass1 / (sin(i)**3)'
        abs_mass2_unc = f'min_mass2 / (sin(i)**3)'
        e_abs_mass1 = ophobning(abs_mass1_unc,varsAndVals,False)
        e_abs_mass2 = ophobning(abs_mass2_unc,varsAndVals,False)

        if make_table:
                names_to_add = ['G_a', 'G_Omega', 'G_w', 'G_i',
                                 'e_G_a', 'e_G_Omega', 'e_G_w', 'e_G_i']
                values_to_add = [a,Omega,w,i, e_a,e_Omega,e_w,e_i]
                for v,n in zip(values_to_add,names_to_add):
                    mt.add_value(v,n,ID)
        

        return abs_mass1, abs_mass2,e_abs_mass1, e_abs_mass2, 
        
    
    if (print_mass ==True):

        T_Innes_path = '/home/lakeclean/Documents/speciale/thiele_innes_elements.txt'
        df = pd.read_csv(T_Innes_path).to_numpy()
        for line in df:
            if line[0] == ID:
                A = line[2]
                B = line[3]
                F = line[4]
                G = line[5]
                e_A = line[6]
                e_B = line[7]
                e_F = line[8]
                e_G = line[9]
                
        if make_table:
                names_to_add = ['G_ATI', 'G_BTI', 'G_FTI', 'G_GTI',
                                 'e_G_ATI', 'e_G_BTI', 'e_G_FTI', 'e_G_GTI']
                values_to_add = [A,B,F,G,e_A,e_B,e_F,e_G]
                for v,n in zip(values_to_add,names_to_add):
                    mt.add_value(v,n,ID)

        if (SB_type == 2):
            abs_mass1, abs_mass2, e_abs_mass1, e_abs_mass2= absolute_mass(k1,k2,e,p, A,B,F,G,e_k1,
                                                                          e_k2,e_e,e_p,e_A,e_B,e_F,e_G)
            min_mass1, min_mass2,e_min_mass1,e_min_mass2 = minimum_mass(k1,k2,e,p,e_k1,e_k2,e_e,e_p)
        
            print(f'Minimum mass 1: {min_mass1}+/- {e_min_mass1} | Minimum mass 2: {min_mass2} +/-{e_min_mass2}')
            print(f'Absolute mass 1: {abs_mass1}+/- {e_abs_mass1} | Absolute mass 2: {abs_mass2} +/-{e_abs_mass2}')
            
            if make_table:
                values_to_add = [abs_mass1, abs_mass2,min_mass1, min_mass2, e_abs_mass1, e_abs_mass2,e_min_mass1,e_min_mass2]
                names_to_add = ['M1','M2','min_M1','min_M2','e_M1','e_M2','e_min_M1','e_min_M2']
                for v,n in zip(values_to_add,names_to_add):
                    mt.add_value(v,n,ID)
        
        if (SB_type == 1):
            abs_mass_func, e_abs_mass_func  = absolute_mass_func(k,e,p,A,B,F,G,e_k,e_e,e_p,e_A,e_B,e_F,e_G)
            min_mass_func,e_min_mass_func = minimum_mass_func(k,e,p,e_k,e_e,e_p)
            print(f'Minimum mass function {min_mass_func}+/-{e_min_mass_func}, Absolute mass function: {abs_mass_func}+/-{e_abs_mass_func}')
            

    ############################## Making table of info: #################################
        
        
        
        '''
        table_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/rv_table.txt'
        if len(glob.glob(table_path))>0:
            pass
            #input(f'The file {table_path} already exist')
            
        if print_mass ==True:
            if SB_type == 2:
                output = f'ID: {ID}'
                output += f'Minimum mass 1: {min_mass1}+/- {e_min_mass1} | Minimum mass 2: {min_mass2} +/-{e_min_mass2}\n'
                output += f'Absolute mass 1: {abs_mass1}+/- {e_abs_mass1} | Absolute mass 2: {abs_mass2} +/-{e_abs_mass2}\n'
            if SB_type == 1:
                output = f'ID: {ID},    Minimum mass function: {min_mass_func}+/-{e_min_mass_func},'
                output += f'Absolute mass function: {abs_mass_func}+/-{e_abs_mass_func}\n'      
        else:
            output = f'ID: {ID} \n'
            
        
        output += ' Date | JD | v_helio | v_bary | rv1 | rv2 | '
        output += 'err_1 | err2 | flux level | \n'

        for i in range(len(epoch_dates)):
            
            #finding a representative flux level for each file:
            flux_level = 0
            for j in range(40):
                try:
                    flux_path = f'/home/lakeclean/Documents/speciale/target_analysis/' + f'{ID}/{epoch_dates[i]}/data/order_{20+j}_raw_spectrum.txt'
                    df = pd.read_csv(flux_path)
                    df = df.to_numpy()
                    flux_level += np.median(df[:,1])
                except:
                    continue
            flux_level = flux_level/40

            #Constructing line for each file:
            output += f'{epoch_dates[i]} | {np.round(epoch_jds[i],2)} | '
            output += f'{np.round(epoch_vhelio[i],2)} | '
            output += f'{np.round(epoch_vbary[i],2)} | {np.round(epoch_rv1s[i],2)} | '
            output += f'{np.round(epoch_rv2s[i],2)} | {np.round(epoch_rv1_errs[i],2)} | '
            output += f'{np.round(epoch_rv2_errs[i],2)} | {np.round(flux_level,4)} \n'

        f = open(table_path,'w')
        f.write(output)
        f.close()
        '''

        

    ################################## Phase Plot: #################################
    if make_phase_plot:
        
        #We use period from fit if possible
        if len(fit_params)>0:
            
            #Plotting the phase plot:
            if SB_type == 1:
                fig,ax = plt.subplots(2,1,sharex = True, height_ratios=[3,1])
                ax[0].plot([0,p],[v0,v0], ls='--',color='black',alpha=0.4,label=f'v0={np.round(v0,2)}km/s')
            if SB_type == 2:
                fig,ax = plt.subplots(3,1,sharex = True, height_ratios=[3,1,1])
                ax[0].plot([0,p],[v01,v01], ls='--',color='black',alpha=0.4,label=f'v01={np.round(v01,2)}km/s')
                ax[0].plot([0,p],[v02,v02], ls='-.',color='black',alpha=0.4,label=f'v02={np.round(v02,2)}km/s')

            fig.subplots_adjust(hspace=0)
            ax[0].set_title(f'Phase plot of ID: {ID}')
            
            ax[0].set_ylabel('radial velocity [km/s]')
            
            ax[0].errorbar(epoch_jds[included1]%p, epoch_rv1s[included1], epoch_rv1_errs[included1],
                   fmt='o',capsize=2,color='r')
            
            ax[0].errorbar(epoch_jds[excluded1]%p, epoch_rv1s[excluded1], epoch_rv1_errs[excluded1],
                   fmt='x',capsize=2,color='grey')
            
            ax[0].errorbar(epoch_jds[included2]%p, epoch_rv2s[included2], epoch_rv2_errs[included2],
                   fmt='o',capsize=2,color='b')
            
            ax[0].errorbar(epoch_jds[excluded2]%p, epoch_rv1s[excluded2], epoch_rv1_errs[excluded2],
                   fmt='x',capsize=2,color='grey')

            #TNG
            ax[0].errorbar(TNG_jds%p, TNG_rv2s, TNG_rv2_errs,
                   fmt='o',capsize=2,color='green')
            ax[0].errorbar(TNG_jds%p, TNG_rv1s, TNG_rv1_errs,
                   fmt='o',capsize=2,color='green')


            
            proxy_time = np.linspace(0,p,1000)

            
            
            if SB_type == 1:
                fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
                ax[0].plot(proxy_time,fit_rvs)
                ax[1].set_xlabel(f'orbital phase jd%{np.round(p,2)} [days]')
                
                #Residuals:
                residual = epoch_rv1s-sb.radial_velocity(epoch_jds%p,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
                ax[1].errorbar(epoch_jds[included1]%p,residual[included1],epoch_rv1_errs[included1],
                               fmt='o',capsize=2,color='b')
                ax[1].errorbar(epoch_jds[excluded1]%p,residual[excluded1],epoch_rv1_errs[excluded1],
                               fmt='x',capsize=2,color='grey')
                ax[1].set_ylim(-res1*np.std(residual[included1]) + res_off1,
                               res_off1 + res1*np.std(residual[included1]))
                
                ax[1].plot([0,p],[0,0]
                   ,ls='--',color='black',alpha=0.4)
                RMS1 = np.sqrt(np.mean(residual[included1]**2))
                ax[1].plot([0,p],[RMS1,RMS1],
                           ls='-.',color='black',alpha=0.4,
                           label=f'RMS: {np.round(RMS1,2)}')
                ax[1].legend()
                
            if SB_type == 2:
                fit_rv1s = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v01)
                ax[0].plot(proxy_time,fit_rv1s)

                fit_rv1s = sb.radial_velocity(proxy_time,k=-k2,e=e,w=w,p=p,t0=t0,v0=v02)
                ax[0].plot(proxy_time,fit_rv1s)
                
                ax[2].set_xlabel(f'orbital phase jd%{np.round(p,2)} [days]')
                
                #Residuals:
                residual1 = epoch_rv1s-sb.radial_velocity(epoch_jds%p,k=k1,e=e,w=w,p=p,t0=t0,v0=v01)
                ax[1].set_ylim(-res1*np.std(residual1[included1]) + res_off1,
                               res_off1 + res1*np.std(residual1[included1]))
                ax[1].errorbar(epoch_jds[included1]%p,residual1[included1],epoch_rv1_errs[included1]
                               ,fmt='o',capsize=2,color='r')
                ax[1].errorbar(epoch_jds[excluded1]%p,residual1[excluded1],epoch_rv1_errs[excluded1]
                               ,fmt='x',capsize=2,color='grey')
                ax[1].plot([0,p],[0,0]
                   ,ls='--',color='black',alpha=0.4)
                
                RMS1 = np.sqrt(np.mean(residual1[included1]**2))
                ax[1].plot([0,p],[RMS1,RMS1],
                           ls='-.',color='black',alpha=0.4,
                           label=f'RMS: {np.round(RMS1,2)}')
                
                residual2 = epoch_rv2s-sb.radial_velocity(epoch_jds%p,k=-k2,e=e,w=w,p=p,t0=t0,v0=v02)
                ax[2].set_ylim(-res2*np.std(residual2[included2]) + res_off2,
                               res_off2+ res2*np.std(residual2[included2]))
                ax[2].errorbar(epoch_jds[included2]%p,residual2[included2],epoch_rv2_errs[included2],
                               fmt='o',capsize=2,color='b')
                ax[2].errorbar(epoch_jds[excluded2]%p,residual2[excluded2],epoch_rv2_errs[excluded2],
                               fmt='x',capsize=2,color='grey')
                ax[2].plot([0,p],[0,0]
                   ,ls='--',color='black',alpha=0.4)
                RMS2 = np.sqrt(np.mean(residual2[included1]**2))
                ax[2].plot([0,p],[RMS2,RMS2],
                           ls='-.',color='black',alpha=0.4,
                           label=f'RMS: {np.round(RMS2,2)}')
                ax[1].legend()
                ax[2].legend()
                
        #We use the orbital period as given:
        else:
            fig, ax = plt.subplots()
            ax.set_title(f'Phase plot of ID: {ID}')
            ax.set_ylabel('radial velocity [km/s]')
            
            ax.errorbar(np.array(epoch_jds)%orbital_period, epoch_rv1s, epoch_rv1_errs,
                    fmt='o',capsize=2,color='r')

            ax.errorbar(np.array(epoch_jds)%orbital_period, epoch_rv2s, epoch_rv2_errs,
                    fmt='o',capsize=2,color='b')
            ax.set_xlabel(f'orbital phase jd%{np.round(orbital_period,2)} [days]')


        plot_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/'
        if save_plot: fig.savefig(plot_path+f"rv_phase_{ID}.pdf",
                                   dpi='figure', format='pdf')
        if show_plot: plt.show()
        plt.close()


    ############################# River Plot: #######################################
    if make_river_plot:
        path = '/home/lakeclean/Documents/speciale/target_analysis/'

        #We construct a list of the 30th order for all the times of the star
        date_len = len(path + ID)+1
        folder_dates = glob.glob(path + ID + '/*')
        print(path + ID + '/*')

        def sorter(x):
            return Time(x[date_len:]).mjd

        folder_dates = sorted(folder_dates,key=sorter)

        rv_region = 401 #The size of the rv region of the BF
        num_dates = len(folder_dates) #Number of spectra
        nr_order_in_mean = 40 #The number of orders considered in mean
        
        dates = []
        bfs = np.zeros(shape=(rv_region,num_dates))
        smoothed = np.zeros(shape=(num_dates,rv_region))
        rvs = np.zeros(rv_region)
        proxy_smoothed = np.zeros(shape=(num_dates,rv_region)) #The shifted wavelengths

        
        for i,folder_date in enumerate(folder_dates):
            dates.append(Time(folder_date[date_len:]).jd)
            for j in range(nr_order_in_mean):
                try:
                    df = pd.read_csv(folder_date + f'/data/order_{j+20}_broadening_function.txt')
                except:
                    print(folder_date+' could not be found. If 2024-07-13T00:26:25.672 then its a bad spec')
                    continue
                
                df = df.to_numpy()
                bfs[:,i] = df[:,1]
                rvs = df[:,0]
                smoothed[i,:] += df[:,2] #Artihmetic mean
                #smoothed[i,:] *= df[:,2] #Geometric mean

        #smoothed = smoothed**(1/nr_order_in_mean) #Geometric mean
        smoothed = smoothed/nr_order_in_mean #Artihmetic mean

        dates = np.array(dates) - 2457000 # Time the way Frank likes

        fig,ax = plt.subplots()
        levels = np.linspace(scale_river[0],scale_river[1],scale_river[2])

        #Shifting the broadening funcitons by the barycentric correction
        
        for i in range(num_dates):
            shift = int(epoch_vbary[i])
            for j in np.arange(shift+40,rv_region-40,1):
                proxy_smoothed[i][j-shift] = smoothed[i][j]

                    
                    
        print(proxy_smoothed)
        cs = ax.contourf(rvs,dates,proxy_smoothed,levels, cmap='RdGy')
        fig.colorbar(cs)
        ax.set_xlabel('Radial Velocity [km/s]')
        ax.set_ylabel(f'JD - 2457000 [days]')
        ax.set_title(f'River plot of ID: {ID}')

        if len(fit_params)>0:
            #ax.set_xlabel(f'orbital phase jd%{np.round(p,2)} [days]')
            proxy_time = np.linspace(min(epoch_jds),max(epoch_jds),1000)
        
            if SB_type == 1:
                fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
                ax.plot(fit_rvs,proxy_time)
                ax.plot([v0,v0],[min(epoch_jds),max(epoch_jds)],color='green',alpha=0.4,ls='--',
                    label=f'v0 = {np.round(v0,2)}')
                ax.set_xlim(v0-100,v0+100)
                
            if SB_type == 2:
                fit_rv1s = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v01)
                ax.plot(fit_rv1s,proxy_time)

                fit_rv1s = sb.radial_velocity(proxy_time,k=-k2,e=e,w=w,p=p,t0=t0,v0=v02)
                ax.plot(fit_rv1s,proxy_time)
                
                ax.plot([v01,v01],[min(epoch_jds),max(epoch_jds)],color='blue',alpha=0.4,ls='--',
                    label=f'v01 = {np.round(v01,2)}')
                ax.plot([v02,v02],[min(epoch_jds),max(epoch_jds)],color='green',alpha=0.4,ls='--',
                    label=f'v02 = {np.round(v02,2)}')
                ax.set_xlim(v01-100,v01+100)
        
        ax.legend()
        
        
        plot_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/'
        if save_plot: fig.savefig(plot_path+f"river_plot_{ID}.pdf",
                                   dpi='figure', format='pdf')
        if show_plot: plt.show()
        plt.close()


    
        
        



#KIC-12317678
if False:
    plot_rv_time('KIC12317678',fit_params=[18,26,0.3,100,82,-41],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.01,100],
                 make_table=True,print_mass = True,exclude_points=12,
                 res1 = 3,
                 find_rv=True)



#KIC-9693187
if False:
    plot_rv_time('KIC9693187',fit_params=[29,26,0.5,50,104,-9],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,print_mass = True,exclude_points=10,
                 res1=3,
                 find_rv=True)

#KIC-4914923
if False:
    plot_rv_time('KIC4914923',fit_params=[15,0.2,105,99,-24],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,print_mass = True,exclude_points=8,
                 res1=8,
                 find_rv=True)


#KIC-9025370
if False:
    plot_rv_time('KIC9025370',fit_params=[16,16,0.271,200,239,-14],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=False, scale_river=[-0.0005,0.14,100],
                 make_table=True,print_mass=True,exclude_points=12,
                 res1=4,res_off1=-0.04,res2=4,res_off2=0.03,
                 find_rv=False)


#KIC-10454113
if True:
    plot_rv_time('KIC10454113',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=False, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=False)

'''
#KIC4457331
if True:
    plot_rv_time('KIC4457331',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)

#EPIC-246696804
if True:
    plot_rv_time('EPIC-246696804',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)
    
#EPIC-212617037
if True:
    plot_rv_time('EPIC-212617037',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)
#EPIC-249570007
if True:
    plot_rv_time('EPIC-249570007',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)

#EPIC-230748783
if True:
    plot_rv_time('EPIC-230748783',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.12,100],
                 make_table=True,
                 find_rv=True)


#EPIC-236224056
if True:
    plot_rv_time('EPIC-236224056',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,
                 find_rv=True)

#KIC4260884
if True:
    plot_rv_time('KIC4260884',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,
                 find_rv=True)

#KIC9652971
if True:
    plot_rv_time('KIC9652971',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.001,0.08,100],
                 make_table=True,
                 find_rv=True)

#KIC4260884
if True:
    plot_rv_time('HD208139',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=False,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,
                 find_rv=True)

'''

#Really good guesses:
'''
#KIC-12317678
full orbit is observed
Parameters([('k1', <Parameter 'k1', value=18.146551986461894, bounds=[0.0:inf]>), ('e', <Parameter 'e', value=0.30842419358606826, bounds=[0.0:1.0]>),
('w', <Parameter 'w', value=278.733967684972, bounds=[0.0:360.0]>), ('p', <Parameter 'p', value=80.0608252327795, bounds=[-inf:inf]>),
('t0', <Parameter 't0', value=-68.1503729095733, bounds=[-inf:inf]>), ('v0_1', <Parameter 'v0_1', value=-41.08561830926434, bounds=[-inf:inf]>)])

#KIC-9693187
full orbit is observed
Parameters([('k1', <Parameter 'k1', value=29.189674847826193, bounds=[0.0:inf]>), ('k2', <Parameter 'k2', value=26.29127616755267, bounds=[0.0:inf]>),
('e', <Parameter 'e', value=0.4493994177095235, bounds=[0.0:1.0]>), ('w', <Parameter 'w', value=44.437936702501176, bounds=[0.0:360.0]>),
('p', <Parameter 'p', value=104.10440279140843, bounds=[-inf:inf]>), ('t0', <Parameter 't0', value=-258.5980636542009, bounds=[-inf:inf]>),
('v0_1', <Parameter 'v0_1', value=-9.622957650745287, bounds=[-inf:inf]>)])


#KIC-10454113
Looks like a really long period orbit. A straight line would fit well. It is increasing in period though slowly.
Changes a few hundred meters per second over a two hundred day observation time span
'''

'''
#KIC-4914923
Parameters([('k1', <Parameter 'k1', value=15.451681911047931, bounds=[0.0:inf]>), ('e', <Parameter 'e', value=0.20733061037149964, bounds=[0.0:1.0]>),
('w', <Parameter 'w', value=105.11050703896773, bounds=[0.0:360.0]>), ('p', <Parameter 'p', value=99.20286889338607, bounds=[-inf:inf]>),
('t0', <Parameter 't0', value=-166.2733050942353, bounds=[-inf:inf]>), ('v0_1', <Parameter 'v0_1', value=-24.398265475664452, bounds=[-inf:inf]>)])
'''
'''

#KIC4457331
All of the spectra have almost duplicates so actually not that many observations. Looks like it is peaking in orbit but hard to tell.
The period must be very high, since there is not much change in hundred days.

#EPIC-246696804
Too few points to fit. Points have high error. the orbit could be flat. 

#EPIC-212617037
Too few points to fit.Points have high error. the orbit could be flat. 

#EPIC-249570007
Too few points to fit. Apears to move in orbit. Further observations should incouraged to confirm

#KIC-9025370 
Doesn't fit very well. Issue with the first point. Does show clear orbit. The full orbit is not resolved however.
Parameters([('k1', <Parameter 'k1', value=15.711641164783249 +/- 6.94, bounds=[0.0:inf]>), ('k2', <Parameter 'k2', value=16.480683912789605 +/- 7.25, bounds=[0.0:inf]>),
('e', <Parameter 'e', value=0.4710733560091602 +/- 0.23, bounds=[0.0:1.0]>), ('w', <Parameter 'w', value=0.001668171614703784 +/- 11.8, bounds=[0.0:360.0]>),
('p', <Parameter 'p', value=196.5017765473483 +/- 1.12, bounds=[-inf:inf]>), ('t0', <Parameter 't0', value=1044.634345374434 +/- 341, bounds=[-inf:inf]>),
('v0_1', <Parameter 'v0_1', value=-13.551532635800074 +/- 0.827, bounds=[-inf:inf]>)])


#EPIC-230748783
Too few points to fit, but probably a change in radial valocity. The rv is very well determined, but the change in radial velocity is very low. Maybe low mass companion?


#EPIC-236224056
Too few points, but probably a change in radial valocity. The rv is very well determined, but the change in radial velocity is very low. Maybe low mass companion?

#KIC4260884
Does vary several km/s but basically looks like an incresaing straight line. Period must be at least and quite likely much higher than 100days.

#KIC9652971
Looks like two straight flat lines. Period must be immense.

'''
























    
