import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd
from astropy.modeling import models, fitting
import sboppy as sb
from astropy.time import Time
import shazam


def plot_rv_time(ID,fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=False, report_fit=False,
                 make_phase_plot=False,make_river_plot=False,scale_river=[-0.0005,0.01,100],
                 make_table=False,print_mass=False,plot_offset=False,exclude_points=5):
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
        

    Returns:
        - resulting_fit  : if fit_params not empty returns array of fit to rv

    Note:
        - fit_params SB1 : [K_guess, e_guess, w_guess,period_guess, v0_guess]
        - fit_params SB2 : [K1_guess,K2_guess, e_guess, w_guess,period_guess, v0_guess]
    '''
    
    path = f'/home/lakeclean/Documents/speciale/order_file_log.txt'
    lines= open(path).read().split('\n')
    all_IDs, all_dates, all_vhelios,files= [], [], [], []
    all_vbary = {}
    for line in lines[:-1]:
        line = line.split(',')
        if line[1].strip() == 'science':
            if line[3].strip() not in all_dates:
                all_IDs.append(line[0].strip())
                all_dates.append(line[3].strip())
                all_vhelios.append(float(line[5].strip()))
                files.append(line[2].strip())
                all_vbary[line[3].strip()] = float(line[-1].strip())/1000 #correcting from m/s to km/s
        
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
        
        #Finding the offset from the median of the best region:
        offset1s[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
        offset2s[i,:] = np.median(vrad2s[i][limits[0]:limits[1]]) - vrad2s[i]
            

    #The offset for each order based on every file  
    offset1s_median = np.median(offset1s,axis=0)
    offset2s_median = np.median(offset2s,axis=0)

    if plot_offset:
        fig,ax = plt.subplots()
        for i in range(len(vrad1s[:,0])):
            ax.scatter(range(len(offset2s[i,:])),offset2s[i,:])
        plt.show()
        plt.close()
    
    weight1s = np.std(offset1s,axis=0)
    weight2s = np.std(offset2s,axis=0)

    epoch_rv1s = []
    epoch_rv2s = []
    epoch_rv1_errs = []
    epoch_rv2_errs = []
    epoch_jds = []
    epoch_dates = []
    epoch_vbary = []
    epoch_vhelio = []
    for i,all_ID in enumerate(all_IDs):
        if not ID == all_ID:
            continue
        epoch_dates.append(all_dates[i])
        epoch_vbary.append(all_vbary[all_dates[i]])
        epoch_vhelio.append(all_vhelios[i])

        #Correcting vrads for NOT wavelength issue
        corr_vrad1s = vrad1s[i]+offset1s_median
        corr_vrad2s = vrad2s[i]+offset2s_median

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
    if ID == 'KIC-12317678':
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
                if IDline.split(',')[0][11:].strip(' ') == 'KIC-12317678':
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
        print(distances)
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
            

    else:
        ax[0].errorbar(epoch_jds, epoch_rv1s, epoch_rv1_errs,
                   fmt='o',capsize=2,color='r')

        ax[0].errorbar(epoch_jds, epoch_rv2s, epoch_rv2_errs,
                   fmt='o',capsize=2,color='b')



    
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
        residual = sb.radial_velocity(epoch_jds,k=k,e=e,w=w,p=p,t0=t0,v0=v0)-epoch_rv1s
        ax[1].errorbar(epoch_jds,residual,epoch_rv1_errs,
               fmt='o',capsize=2,color='r')
        
        ax[1].plot([min(epoch_jds),max(epoch_jds)],[0,0]
                   ,ls='--',color='black',alpha=0.4)
        
        

    if (len(fit_params)> 0) and (SB_type == 2):
        ##################### SB2 ######################

        #Fitting
        K1_guess, K2_guess, e_guess, w_guess, period_guess, v0_guess = fit_params

        rvs = [epoch_rv1s[included1],epoch_rv2s[included2],epoch_rv1_errs[included1],epoch_rv2_errs[included2]]
        jds = [epoch_jds[included1],epoch_jds[included2]]
        fit = sb.fit_radvel_SB2(jds,rvs,k=[K1_guess,K2_guess],e=e_guess,
                                w=w_guess,p=period_guess,v0=v0_guess)

        k1 = fit.params['k1'].value
        k2 = fit.params['k2'].value
        e = fit.params['e'].value
        w = fit.params['w'].value
        p = fit.params['p'].value
        t0 = fit.params['t0'].value
        v0 = fit.params['v0_1'].value

        if report_fit: print(fit.params)
            
        proxy_time = np.linspace(min(epoch_jds),max(epoch_jds),1000)

        fit_rv1s = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)
        ax[0].plot(proxy_time,fit_rv1s,label='fit')

        fit_rv2s = sb.radial_velocity(proxy_time,k=-k2,e=e,w=w,p=p,t0=t0,v0=v0)
        ax[0].plot(proxy_time,fit_rv2s,label='fit')
        
        ax[0].plot([min(epoch_jds),max(epoch_jds)],[v0,v0],
                   ls='--',color='black',alpha=0.4,label=f'v0={np.round(v0,2)}km/s')
        ax[0].legend()
        

        #Residuals:
        residual1 = sb.radial_velocity(epoch_jds,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)-epoch_rv1s
        ax[1].errorbar(epoch_jds,residual1,epoch_rv1_errs,
               fmt='o',capsize=2,color='r')
        
        residual2 = sb.radial_velocity(epoch_jds,k=-k2,e=e,w=w,p=p,t0=t0,v0=v0)-epoch_rv2s
        ax[2].errorbar(epoch_jds,residual2,epoch_rv2_errs,
               fmt='o',capsize=2,color='b')
        
        ax[1].plot([min(epoch_jds),max(epoch_jds)],[0,0]
                   ,ls='--',color='black',alpha=0.4)
        
        ax[2].plot([min(epoch_jds),max(epoch_jds)],[0,0]
                   ,ls='--',color='black',alpha=0.4)

        

        

    plot_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/'
    if save_plot: fig.savefig(plot_path+f"rv_time_{ID}.svg",
                                   dpi='figure', format='svg')
                  
    if show_plot: plt.show()
    plt.close()

    ############################## Minimum mass estimate: ##############################

    def minimum_mass(k1,k2,e,p):
        sun_mass = 1.988 * 10**30 #kg
        G  =6.674*10**(-11) #N m^2 / kg ^2
        min_mass1 = (1+k1/k2)**2 * p*24*60*60 * (k2*1000)**3 * (1-e**2)**(3/2) / (2*np.pi*G)
        min_mass2 = (1+k2/k1)**2 * p*24*60*60 * (k1*1000)**3 * (1-e**2)**(3/2) / (2*np.pi*G)
        return min_mass1/sun_mass, min_mass2/sun_mass
    
    if (print_mass ==True) and (SB_type == 2) :
        min_mass1, min_mass2 = minimum_mass(k1,k2,e,p)
        print(f'The minimum masses of the two components are: {min_mass1} and {min_mass2}')

    

    ############################## Making table of info: #################################

    #importing raw spectrum:
    
    if make_table:
        table_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/rv_table.txt'
        if len(glob.glob(table_path))>0:
            input(f'The file {table_path} already exist')
            
        if print_mass ==True:
            output = f'ID: {ID},    Mass estiamte: {min_mass1} {min_mass2}\n'
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

    ################################## Phase Plot: #################################
    if make_phase_plot:
        
        #We use period from fit if possible
        if len(fit_params)>0:
            
            #Plotting the phase plot:
            if SB_type == 1:
                fig,ax = plt.subplots(2,1,sharex = True, height_ratios=[3,1])
            if SB_type == 2:
                fig,ax = plt.subplots(3,1,sharex = True, height_ratios=[3,1,1])

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
            
            proxy_time = np.linspace(0,p,1000)

            
            ax[0].plot([0,p],[v0,v0], ls='--',color='black',alpha=0.4,label=f'v0={np.round(v0,2)}km/s')
            
            if SB_type == 1:
                fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
                ax[0].plot(proxy_time,fit_rvs)
                ax[1].set_xlabel(f'orbital phase jd%{np.round(p,2)} [days]')
                
                #Residuals:
                residual = sb.radial_velocity(np.array(epoch_jds)%p,k=k,e=e,w=w,p=p,t0=t0,v0=v0)-epoch_rv1s
                ax[1].errorbar(np.array(epoch_jds)%p,residual,epoch_rv1_errs,fmt='o',capsize=2,color='b')
                
                ax[1].plot([0,p],[0,0]
                   ,ls='--',color='black',alpha=0.4)
                
            if SB_type == 2:
                fit_rv1s = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)
                ax[0].plot(proxy_time,fit_rv1s)

                fit_rv1s = sb.radial_velocity(proxy_time,k=-k2,e=e,w=w,p=p,t0=t0,v0=v0)
                ax[0].plot(proxy_time,fit_rv1s)
                
                ax[2].set_xlabel(f'orbital phase jd%{np.round(p,2)} [days]')
                
                #Residuals:
                residual1 = sb.radial_velocity(np.array(epoch_jds)%p,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)-epoch_rv1s
                ax[1].errorbar(np.array(epoch_jds)%p,residual1,epoch_rv1_errs,fmt='o',capsize=2,color='r')
                ax[1].plot([0,p],[0,0]
                   ,ls='--',color='black',alpha=0.4)
                
                residual2 = sb.radial_velocity(np.array(epoch_jds)%p,k=-k2,e=e,w=w,p=p,t0=t0,v0=v0)-epoch_rv2s
                ax[2].errorbar(np.array(epoch_jds)%p,residual2,epoch_rv2_errs,fmt='o',capsize=2,color='b')
                ax[2].plot([0,p],[0,0]
                   ,ls='--',color='black',alpha=0.4)
                
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
        if save_plot: fig.savefig(plot_path+f"rv_phase_{ID}.svg",
                                   dpi='figure', format='svg')
        if show_plot: plt.show()
        plt.close()


    ############################# River Plot: #######################################
    if make_river_plot:
        path = '/home/lakeclean/Documents/speciale/target_analysis/'

        #We construct a list of the 30th order for all the times of the star
        date_len = len(path + ID)+1
        folder_dates = glob.glob(path + ID + '/*')

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

                    
                    
            
        cs = ax.contourf(rvs,dates,proxy_smoothed,levels, cmap='RdGy')
        fig.colorbar(cs)
        ax.set_xlabel('Radial Velocity [km/s]')
        ax.set_ylabel(f'JD - 2457000 [days]')
        ax.set_title(f'River plot of ID: {ID}')

        if len(fit_params)>0:
            ax.set_xlabel(f'orbital phase jd%{np.round(p,2)} [days]')
            proxy_time = np.linspace(min(epoch_jds),max(epoch_jds),1000)
            ax.plot([v0,v0],[min(epoch_jds),max(epoch_jds)],color='green',alpha=0.4,ls='--',
                    label=f'v0 = {np.round(v0,2)}')
            if SB_type == 1:
                fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
                ax.plot(fit_rvs,proxy_time)
                
            if SB_type == 2:
                fit_rv1s = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)
                ax.plot(fit_rv1s,proxy_time)

                fit_rv1s = sb.radial_velocity(proxy_time,k=-k2,e=e,w=w,p=p,t0=t0,v0=v0)
                ax.plot(fit_rv1s,proxy_time)
        ax.legend()
        
        
        plot_path = f'/home/lakeclean/Documents/speciale/rv_plots/{ID}/'
        if save_plot: fig.savefig(plot_path+f"river_plot_{ID}.svg",
                                   dpi='figure', format='svg')
        if show_plot: plt.show()
        plt.close()
        
        



#KIC-12317678
if False:
    plot_rv_time('KIC-12317678',fit_params=[18,26,0.3,100,82,-41],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.01,100],
                 make_table=True,print_mass = True,exclude_points=8)



#KIC-9693187
if False:
    plot_rv_time('KIC-9693187',fit_params=[29,26,0.5,50,104,-9],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True)

#KIC-4914923
if False:
    plot_rv_time('KIC-4914923',fit_params=[15,0.2,105,99,-24],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True)


#KIC-9025370
if False:
    plot_rv_time('KIC-9025370',fit_params=[16,16,0.271,200,239,-14],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,print_mass=True,exclude_points=6)

#KIC-10454113
if False:
    plot_rv_time('KIC-10454113',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True)


#KIC4457331
if False:
    plot_rv_time('KIC4457331',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True)

#EPIC-246696804
if False:
    plot_rv_time('EPIC-246696804',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True)
    
#EPIC-212617037
if False:
    plot_rv_time('EPIC-212617037',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True)
#EPIC-249570007
if False:
    plot_rv_time('EPIC-249570007',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True)

#EPIC-230748783
if False:
    plot_rv_time('EPIC-230748783',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.12,100],
                 make_table=True)


#EPIC-236224056
if False:
    plot_rv_time('EPIC-236224056',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True)

#KIC4260884
if False:
    plot_rv_time('KIC4260884',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True)

#KIC9652971
if False:
    plot_rv_time('KIC9652971',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.001,0.08,100],
                 make_table=True)



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


#KIC-4914923
Parameters([('k1', <Parameter 'k1', value=15.451681911047931, bounds=[0.0:inf]>), ('e', <Parameter 'e', value=0.20733061037149964, bounds=[0.0:1.0]>),
('w', <Parameter 'w', value=105.11050703896773, bounds=[0.0:360.0]>), ('p', <Parameter 'p', value=99.20286889338607, bounds=[-inf:inf]>),
('t0', <Parameter 't0', value=-166.2733050942353, bounds=[-inf:inf]>), ('v0_1', <Parameter 'v0_1', value=-24.398265475664452, bounds=[-inf:inf]>)])


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
























    
