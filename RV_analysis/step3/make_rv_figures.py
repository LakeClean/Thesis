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
master_path = '/usr/users/au662080'

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

    
    df = pd.read_csv(f'{master_path}/Speciale/data/rv_data/NOT_{ID}.txt')
    rv_names = ['jd','date','rv1','rv2','e_rv1','e_rv2','vbary']
    epoch_rv1s = df['rv1'].to_numpy()
    epoch_rv2s = df['rv2'].to_numpy()
    epoch_rv1_errs = df['e_rv1'].to_numpy()
    epoch_rv2_errs = df['e_rv2'].to_numpy()
    epoch_jds = df['jd'].to_numpy()- 2457000 #The correction Frank likes
    epoch_dates = df['date'].to_numpy()
    epoch_vbary = df['vbary'].to_numpy()

    try:
        df = pd.read_csv(f'{master_path}/Speciale/data/rv_data/NOT_{ID}.txt')
        NOT_rv1s = df['rv1'].to_numpy()
        NOT_rv2s = df['rv2'].to_numpy()
        NOT_rv1_errs = df['e_rv1'].to_numpy()
        NOT_rv2_errs = df['e_rv2'].to_numpy()
        NOT_jds = df['jd'].to_numpy()- 2457000 #The correction Frank likes
        NOT_dates = df['date'].to_numpy()
        NOT_vbary = df['vbary'].to_numpy()
    except:
        pass

    try:
        df = pd.read_csv(f'{master_path}/Speciale/data/rv_data/TNG_{ID}.txt')
        TNG_rv1s = df['rv1'].to_numpy()
        TNG_rv2s = df['rv2'].to_numpy()
        TNG_rv1_errs = df['e_rv1'].to_numpy()
        TNG_rv2_errs = df['e_rv2'].to_numpy()
        TNG_jds = df['jd'].to_numpy() - 2457000 #The correction Frank likes
        TNG_dates = df['date'].to_numpy()
        TNG_vbary = df['vbary'].to_numpy()

        '''
        epoch_rv1s = np.append(epoch_rv1s,TNG_rv1s,axis=0)
        epoch_rv2s = np.append(epoch_rv2s,TNG_rv2s,axis=0)
        epoch_rv1_errs = np.append(epoch_rv1_errs,TNG_rv1_errs,axis=0)
        epoch_rv2_errs = np.append(epoch_rv2_errs,TNG_rv2_errs,axis=0)
        epoch_jds = np.append(epoch_jds,TNG_jds,axis=0)
        epoch_dates = np.append(epoch_dates,TNG_dates,axis=0)
        epoch_vbary = np.append(epoch_vbary,TNG_vbary,axis=0)
        '''
        
    except:
        pass

    try:
        df = pd.read_csv(f'{master_path}/Speciale/data/rv_data/NOT_old_HIRES_{ID}.txt')
        NOT_old_HIRES_rv1s = df['rv1'].to_numpy()
        NOT_old_HIRES_rv2s = df['rv2'].to_numpy()
        NOT_old_HIRES_rv1_errs = df['e_rv1'].to_numpy()
        NOT_old_HIRES_rv2_errs = df['e_rv2'].to_numpy()
        NOT_old_HIRES_jds = df['jd'].to_numpy()- 2457000 #The correction Frank likes
        NOT_old_HIRES_dates = df['date'].to_numpy()
        NOT_old_HIRES_vbary = df['vbary'].to_numpy()

        if True:
            epoch_rv1s = np.append(epoch_rv1s,NOT_old_HIRES_rv1s,axis=0)
            epoch_rv2s = np.append(epoch_rv2s,NOT_old_HIRES_rv2s,axis=0)
            epoch_rv1_errs = np.append(epoch_rv1_errs,NOT_old_HIRES_rv1_errs,axis=0)
            epoch_rv2_errs = np.append(epoch_rv2_errs,NOT_old_HIRES_rv2_errs,axis=0)
            epoch_jds = np.append(epoch_jds,NOT_old_HIRES_jds,axis=0)
            epoch_dates = np.append(epoch_dates,NOT_old_HIRES_dates,axis=0)
            epoch_vbary = np.append(epoch_vbary,NOT_old_HIRES_vbary,axis=0)
        
    except:
        pass

    try:
        df = pd.read_csv(f'{master_path}/Speciale/data/rv_data/NOT_old_LOWRES_{ID}.txt')
        NOT_old_LOWRES_rv1s = df['rv1'].to_numpy()
        NOT_old_LOWRES_rv2s = df['rv2'].to_numpy()
        NOT_old_LOWRES_rv1_errs = df['e_rv1'].to_numpy()
        NOT_old_LOWRES_rv2_errs = df['e_rv2'].to_numpy()
        NOT_old_LOWRES_jds = df['jd'].to_numpy()- 2457000 #The correction Frank likes
        NOT_old_LOWRES_dates = df['date'].to_numpy()
        NOT_old_LOWRES_vbary = df['vbary'].to_numpy()

        if True:
            epoch_rv1s = np.append(epoch_rv1s,NOT_old_LOWRES_rv1s,axis=0)
            epoch_rv2s = np.append(epoch_rv2s,NOT_old_LOWRES_rv2s,axis=0)
            epoch_rv1_errs = np.append(epoch_rv1_errs,NOT_old_LOWRES_rv1_errs,axis=0)
            epoch_rv2_errs = np.append(epoch_rv2_errs,NOT_old_LOWRES_rv2_errs,axis=0)
            epoch_jds = np.append(epoch_jds,NOT_old_LOWRES_jds,axis=0)
            epoch_dates = np.append(epoch_dates,NOT_old_LOWRES_dates,axis=0)
            epoch_vbary = np.append(epoch_vbary,NOT_old_LOWRES_vbary,axis=0)
        
    except:
        pass

    try:
        df = pd.read_csv(f'{master_path}/Speciale/data/rv_data/KECK_{ID}.txt')
        KECK_vbary = df['vbary'].to_numpy()
        KECK_rv1s = df['rv1'].to_numpy() - KECK_vbary
        KECK_rv2s = df['rv2'].to_numpy() - KECK_vbary
        KECK_rv1_errs = df['e_rv1'].to_numpy()
        KECK_rv2_errs = df['e_rv2'].to_numpy()
        KECK_jds = df['jd'].to_numpy() - 2457000 #The correction Frank likes
        KECK_dates = df['date'].to_numpy()
        '''
        epoch_rv1s = np.append(epoch_rv1s,KECK_rv1s,axis=0)
        epoch_rv2s = np.append(epoch_rv2s,KECK_rv2s,axis=0)
        epoch_rv1_errs = np.append(epoch_rv1_errs,KECK_rv1_errs,axis=0)
        epoch_rv2_errs = np.append(epoch_rv2_errs,KECK_rv2_errs,axis=0)
        epoch_jds = np.append(epoch_jds,KECK_jds,axis=0)
        epoch_dates = np.append(epoch_dates,KECK_dates,axis=0)
        epoch_vbary = np.append(epoch_vbary,KECK_vbary,axis=0)
        '''
        
    except:
        pass


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

        try:
            ax[0].errorbar(TNG_jds, TNG_rv1s, TNG_rv1_errs,
                       fmt='o',capsize=2,color='green')
            ax[0].errorbar(TNG_jds, TNG_rv2s, TNG_rv2_errs,
                       fmt='o',capsize=2,color='green',label='TNG')
        except:
            pass
        
        try:
            ax[0].errorbar(NOT_old_LOWRES_jds, NOT_old_LOWRES_rv1s, NOT_old_LOWRES_rv1_errs,
                       fmt='o',capsize=2,color='purple')
            ax[0].errorbar(NOT_old_LOWRES_jds, NOT_old_LOWRES_rv2s, NOT_old_LOWRES_rv2_errs,
                       fmt='o',capsize=2,color='purple')
        except:
            pass
        
        try:
            ax[0].errorbar(NOT_old_HIRES_jds, NOT_old_HIRES_rv1s, NOT_old_HIRES_rv1_errs,
                       fmt='o',capsize=2,color='purple')
            ax[0].errorbar(NOT_old_HIRES_jds, NOT_old_HIRES_rv2s, NOT_old_HIRES_rv2_errs,
                       fmt='o',capsize=2,color='purple',label='old NOT')
        except:
            pass

        try:
            pass
            ax[0].errorbar(KECK_jds, KECK_rv1s, KECK_rv1_errs,
                       fmt='o',capsize=2,color='pink')
            ax[0].errorbar(KECK_jds, KECK_rv2s, KECK_rv2_errs,
                       fmt='o',capsize=2,color='pink',label='KECK')
        except:
            pass


        #Random stuff:
        if True:
            ax[0].errorbar(V_litt_time, V_litt, e_V_litt,
                       fmt='o',capsize=2,color='red',label='SIMBAD: GAIA/APOGEE')
            t2, rv2 = -1190.4594305553474,-32.545914715462075
            ax[0].scatter(t2,rv2,
                          label='2.dary component', color='black')

            
            #extra_times = ['2010-07-08 02:35:07','2010-09-01 23:15:06','2011-07-12 22:03:22',
            #                   '2011-07-19 03:54:56','2011-08-10 03:09:20','2013-06-15 01:29:45',
            #                   '2013-09-09 22:08:58']
            #for extra_time in extra_times:
            #    ax[0].vlines(Time(extra_time).jd-2457000,-40,20,ls='--',color='black')
                
            

    else:
        ax[0].errorbar(epoch_jds, epoch_rv1s, epoch_rv1_errs,
                   fmt='o',capsize=2,color='r')

        ax[0].errorbar(epoch_jds, epoch_rv2s, epoch_rv2_errs,
                   fmt='o',capsize=2,color='b')
        try:
            ax[0].errorbar(TNG_jds, TNG_rv1s, TNG_rv1_errs,
                       fmt='o',capsize=2,color='green')
            ax[0].errorbar(TNG_jds, TNG_rv2s, TNG_rv2_errs,
                       fmt='o',capsize=2,color='green',label='TNG')
        except:
            pass

        try:
            ax[0].errorbar(NOT_old_LOWRES_jds, NOT_old_LOWRES_rv1s, NOT_old_LOWRES_rv1_errs,
                       fmt='o',capsize=2,color='purple')
            ax[0].errorbar(NOT_old_LOWRES_jds, NOT_old_LOWRES_rv2s, NOT_old_LOWRES_rv2_errs,
                       fmt='o',capsize=2,color='purple')
        except:
            pass
        
        try:
            ax[0].errorbar(NOT_old_HIRES_jds, NOT_old_HIRES_rv1s, NOT_old_HIRES_rv1_errs,
                       fmt='o',capsize=2,color='purple')
            ax[0].errorbar(NOT_old_HIRES_jds, NOT_old_HIRES_rv2s, NOT_old_HIRES_rv2_errs,
                       fmt='o',capsize=2,color='purple',label='old NOT')
        except:
            pass

        try:
            ax[0].errorbar(KECK_jds, KECK_rv1s, KECK_rv1_errs,
                       fmt='o',capsize=2,color='pink')
            #ax[0].errorbar(KECK_jds, KECK_rv2s-KECK_vbary, KECK_rv2_errs,
             #          fmt='o',capsize=2,color='pink',label='KECK')
        except:
            pass




        

    ax[0].legend()



    
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
        print(fit.params)
        print(fit.covar)
        
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

        #For KIC10454113:
        '''
        tmax = proxy_time[np.where(max(fit_rvs) == fit_rvs)[0]]
        t2, rv2 = -1190.4594305553474,-32.545914715462075
        print('here:', np.radians(w+180)%(2*np.pi), (np.radians(w+180) + 2*np.pi/p * (t2-tmax))%(2*np.pi) )
        k2 = (rv2- v0) * (e*np.cos(np.radians(w + 180)) -1 ))**(-1)
        print(k2)
        
        fit_rv2 = sb.radial_velocity(proxy_time,k=k2,e=e,w=(180+w),p=p,t0=t0,v0=v0)
        ax[0].plot(proxy_time,fit_rv2,label='fit 2.dary component')
        '''
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

        fit_rv2s = sb.radial_velocity(proxy_time,k=k2,e=e,w=(w-180),p=p,t0=t0,v0=v02)
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
        

        

    plot_path = f'{master_path}/Speciale/data/rv_plots/{ID}/'
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

        T_Innes_path = f'{master_path}/Speciale/data/thiele_innes_elements.txt'
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
            try:
                ax[0].errorbar(TNG_jds%p, TNG_rv2s, TNG_rv2_errs,
                       fmt='o',capsize=2,color='green')
                ax[0].errorbar(TNG_jds%p, TNG_rv1s, TNG_rv1_errs,
                       fmt='o',capsize=2,color='green')
            except:
                pass
            #Old NOT LOWRES
            try:
                ax[0].errorbar(NOT_old_LOWRES_jds%p, NOT_old_LOWRES_rv1s, NOT_old_LOWRES_rv1_errs,
                           fmt='o',capsize=2,color='purple')
                ax[0].errorbar(NOT_old_LOWRES_jds%p, NOT_old_LOWRES_rv2s, NOT_old_LOWRES_rv2_errs,
                           fmt='o',capsize=2,color='purple')
            except:
                pass
            #Old NOT HIRES
            try:
                ax[0].errorbar(NOT_old_HIRES_jds%p, NOT_old_HIRES_rv1s, NOT_old_HIRES_rv1_errs,
                           fmt='o',capsize=2,color='purple')
                ax[0].errorbar(NOT_old_HIRES_jds%p, NOT_old_HIRES_rv2s, NOT_old_HIRES_rv2_errs,
                           fmt='o',capsize=2,color='purple',label='old NOT')
            except:
                pass

            #KECK
            try:
                ax[0].errorbar(KECK_jds%p, KECK_rv1s, KECK_rv1_errs,
                           fmt='o',capsize=2,color='pink',label='KECK')

            except:
                pass


            #Random stuff:
            if False:
                extra_times = ['2010-07-08 02:35:07','2010-09-01 23:15:06','2011-07-12 22:03:22',
                               '2011-07-19 03:54:56','2011-08-10 03:09:20','2013-06-15 01:29:45',
                               '2013-09-09 22:08:58']
                extra_times = ['2010-07-08 04:50:33','2010-09-02 01:22:13','2011-07-21 02:08:08',
                               '2013-06-15 03:50:58','2013-09-10 00:29:24']
                extra_times = ['2011-06-15 23:00:28','2011-08-10 00:57:55']
                extra_times = ['2009-06-05 01:28:56','2009-08-25 21:26:17']
                for extra_time in extra_times:
                    ax[0].vlines((Time(extra_time).jd-2457000)%p,-40,20,ls='--',color='black')
            


            
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


        plot_path = f'{master_path}/Speciale/data/rv_plots/{ID}/'
        if save_plot: fig.savefig(plot_path+f"rv_phase_{ID}.pdf",
                                   dpi='figure', format='pdf')
        if show_plot: plt.show()
        plt.close()


    ############################# River Plot: #######################################
    if make_river_plot:
        path = f'{master_path}/Speciale/data/target_analysis/'

        #Only considering new NOT data. or else it becomes cluttered.
        NOT_path = f'{master_path}/Speciale/data/NOT_order_file_log.txt'
        NOT_dates = pd.read_csv(NOT_path)['date'].to_numpy()

        date_len = len(path + ID)+1
        target_dates = glob.glob(path + ID + '/*')
        folder_dates = []
        
        for epoch_date in epoch_dates:
            for target_date in target_dates:
                if epoch_date == target_date[date_len:]:
                    if epoch_date in NOT_dates:
                        folder_dates.append(path + ID + f'/{epoch_date}')
        
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

        dates = np.array(dates) # It only makes sense to plot phase

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
            proxy_time = np.linspace(min(dates),max(dates),1000)
        
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
                
                ax.plot([v01,v01],[min(dates),max(dates)],color='blue',alpha=0.4,ls='--',
                    label=f'v01 = {np.round(v01,2)}')
                ax.plot([v02,v02],[min(dates),max(dates)],color='green',alpha=0.4,ls='--',
                    label=f'v02 = {np.round(v02,2)}')
                ax.set_xlim(v01-100,v01+100)
        
        ax.legend()
        
        
        plot_path = f'{master_path}/Speciale/data/rv_plots/{ID}/'
        if save_plot: fig.savefig(plot_path+f"river_plot_{ID}.pdf",
                                   dpi='figure', format='pdf')
        if show_plot: plt.show()
        plt.close()


#KIC10454113
if True:
    plot_rv_time('KIC10454113',fit_params=[16,0.7,350,6000,-20],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=False, scale_river=[-0.0005,0.08,100],
                 make_table=True,exclude_points=0)
       
'''
#KIC12317678
if True:
    plot_rv_time('KIC12317678',fit_params=[18,26,0.3,100,82,-41],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=False, scale_river=[-0.0005,0.01,100],
                 make_table=True,print_mass = True,exclude_points=12,
                 res1 = 3)



#KIC9693187
if True:
    plot_rv_time('KIC9693187',fit_params=[29,26,0.9,50,104,-9],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,print_mass = True,exclude_points=10,
                 res1=3)

#KIC4914923
if True:
    plot_rv_time('KIC4914923',fit_params=[15,0.2,105,99,-24],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,print_mass = True,exclude_points=8,
                 res1=8)


#KIC9025370
if True:
    plot_rv_time('KIC9025370',fit_params=[16,16,0.271,200,239,-14],limits=[20,60],SB_type=2,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=False, scale_river=[-0.0005,0.14,100],
                 make_table=True,print_mass=True,exclude_points=12,
                 res1=4,res_off1=-0.04,res2=4,res_off2=0.03)

'''
#KIC10454113
if True:
    plot_rv_time('KIC10454113',fit_params=[16,0.8,100,6000,-20],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=True,make_river_plot=False, scale_river=[-0.0005,0.08,100],
                 make_table=True,exclude_points=0)

'''
#KIC4457331
if True:
    plot_rv_time('KIC4457331',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)



#EPIC246696804
if True:
    plot_rv_time('EPIC246696804',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)
    
#EPIC212617037
if True:
    plot_rv_time('EPIC212617037',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)
#EPIC249570007
if True:
    plot_rv_time('EPIC249570007',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.08,100],
                 make_table=True,
                 find_rv=True)

#EPIC230748783
if True:
    plot_rv_time('EPIC230748783',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.12,100],
                 make_table=True,
                 find_rv=True)


#EPIC236224056
if True:
    plot_rv_time('EPIC236224056',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,
                 find_rv=True)

#KIC4260884
if True:
    plot_rv_time('KIC4260884',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.0005,0.14,100],
                 make_table=True,
                 find_rv=True)

#KIC9652971
if True:
    plot_rv_time('KIC9652971',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
                 make_phase_plot=False,make_river_plot=True, scale_river=[-0.001,0.08,100],
                 make_table=True,
                 find_rv=True)

#HD208139
if True:
    plot_rv_time('HD208139',fit_params=[],limits=[20,60],SB_type=1,
                 orbital_period=100, show_plot=True,save_plot=True, report_fit=True,
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
























    
