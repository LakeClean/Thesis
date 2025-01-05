import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd
from astropy.modeling import models, fitting
import sboppy as sb
from astropy.time import Time
import shazam


'''
Sricpt for constructing "river plot" of the movement of the stars

-We construct a list of the 30th order for all the times of the star
-We extract the bf for these
-we make a contour plot or river plot based on intensity of bf
'''

path = '/home/lakeclean/Documents/speciale/target_analysis/'
ID = 'KIC9652971'


#We construct a list of the 30th order for all the times of the star
def plot_overlay(ID,overlay=False,off=0.002):

    #Finding the vsini:
    bf_fit_path = f'/home/lakeclean/Documents/speciale/rv_data/rv_{ID}.txt'
    df = pd.read_csv(bf_fit_path)
    vsini1s = df['vsini1'].to_numpy()
    vsini2s = df['vsini2'].to_numpy()
    vrad1s = df['rv1'].to_numpy()
    vrad2s = df['rv2'].to_numpy()

    #correcting for vbary:
    def give_bary(epoch_date):
        bary_path = f'/home/lakeclean/Documents/speciale/NOT_order_file_log.txt'
        lines= open(bary_path).read().split('\n')
        for line in lines[:-1]:
            line = line.split(',')
            if line[1].strip() == 'science':
                if line[3].strip() == epoch_date: #Skipping duplicates
                    return float(line[-2].strip())/1000


    
    date_len = len(path + ID)+1
    folder_dates = glob.glob(path + ID + '/*')

    def sorter(x):
        return Time(x[date_len:]).mjd

    folder_dates = sorted(folder_dates,key=sorter)


    num_dates = len(folder_dates)
    dates = []
    bfs = np.zeros(shape=(401,num_dates))
    smoothed = np.zeros(shape=(num_dates,401))
    rvs = np.zeros(shape=(401,num_dates))
    v_barys = np.zeros(num_dates)

    order_sum = 40 #the number of orders that is summed
    for i,folder_date in enumerate(folder_dates):
        dates.append(Time(folder_date[date_len:]).mjd)
        v_barys[i] = give_bary(folder_date[date_len:])
        
        for j in range(order_sum):
            try:
                df = pd.read_csv(folder_date + f'/data/order_{j+20}_broadening_function.txt')
            except:
                print(folder_date+' could not be found. If 2024-07-13T00:26:25.672 then its a bad spec')
                continue
            
            df = df.to_numpy()
            bfs[:,i] = df[:,1]
            rvs[:,i] = df[:,0]
            smoothed[i,:] += df[:,2]

    smoothed = smoothed/order_sum

    mjd_zero = min(dates)

    dates = np.array(dates) - mjd_zero



    fig, ax = plt.subplots()
    offset = min(dates)
    for i in range(len(dates)):
        
        if overlay:
            print(v_barys[i])
            offset = i*off + 0.005
            ax.plot(rvs[:,i]+v_barys[i]-vrad1s[i],smoothed[i,:])
            ax.text(-210,0+offset,np.round(vsini1s[i],2))
            ax.text(-150,0+offset,np.round(vsini2s[i],2))
        else:
            offset = dates[i]*0.02
            ax.plot(rvs[:,i],smoothed[i,:] + offset)
            ax.text(-210,0+offset,np.round(vsini1s[i],2))
            ax.text(-150,0+offset,np.round(vsini2s[i],2))
            
    ax.set_title(f'River plot of ID: {ID}')




    vpath = '/home/lakeclean/Documents/speciale/rv_data'

    df = pd.read_csv(vpath + f'/rv_{ID}.txt')

    #print(df['vsini1'])

    fig, ax  = plt.subplots()
    ax.set_title(f'ID: {ID}')
    ax.errorbar(df['jd'].to_numpy(),df['vsini1'].to_numpy(),df['e_vsini1'].to_numpy()
                ,capsize=2,fmt='o')
    #ax.scatter(df['jd'].to_numpy(),df['rv1'].to_numpy()*0.01+3,color='red',alpha=0.4)
    ax.set_ylabel('vsini km/s')
    ax.set_xlabel('JD - 2457000[days]')









    
    plt.show()

#plot_overlay('KIC9652971',True)
#plot_overlay('EPIC212617037',True) #few points
#plot_overlay('EPIC246696804',True) #few points
#plot_overlay('KIC10454113',True) #many points small variance

#plot_overlay('EPIC249570007',True) #few points, small variance
#plot_overlay('EPIC230748783',True,0.01)#few points, small variance
#plot_overlay('EPIC236224056',True,0.01)#few points, small variance

#plot_overlay('KIC4914923',True)
#plot_overlay('KIC12317678',True)
#plot_overlay('KIC4457331',True,0.01) #few points, small variance
#plot_overlay('KIC4260884',True,0.01) #maybe shows some systematics still small systematics




#Only for #KIC-123...
'''
fig, ax = plt.subplots()
offset = min(dates)


IDlines = open('/home/lakeclean/Documents/speciale/spectra_log_h_readable.txt').read().split('&')
SB2_IDs, SB2_dates, SB_types, vrad1, vrad2 = [], [], [], [], []

for IDline in IDlines[:-1]:
    if IDline.split(',')[0][11:].strip(' ') == 'KIC-12317678':
        print(IDline.split(',')[0][11:].strip(' '))
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
                vrad1.append(float(line[2].split('/')[0].strip(' ')))
                vrad2.append(float(line[2].split('/')[1].strip(' ')))
            else:
                vrad1.append(0)
                vrad2.append(0)

              
for i,SB_type in enumerate(SB_types):
    print(dates[i])
    
    offset = dates[i]*0.01
    ax.plot(rvs[:,i],smoothed[i,:] + offset)
    if SB_type == 1:
        
        fit, model, bfgs = shazam.rotbf_fit(rvs[:,i],smoothed[i,:], 30,60000,1, 5,False)
        ax.plot(rvs[:,i],model + offset)
        
    if SB_type == 2:
        #region #1:
        fit, model, bfgs = shazam.rotbf2_fit(rvs[:,i],smoothed[i,:], 30,60000,1, 5,5,vrad1[i],vrad2[i],0.05,0.05,True, True)

    
    #ax.plot(rvs[:,i],bfgs + offset)
    ax.plot(rvs[:,i],model + offset)
'''


'''
shazam.rotbf2_fit(rvs,bf, rotbf2_fit_fitsize,rotbf2_fit_res,
                                                     rotbf2_fit_smooth, rotbf2_fit_vsini1,
                                                     rotbf2_fit_vsini2,rotbf2_fit_vrad1,
                                                     rotbf2_fit_vrad2,rotbf2_fit_ampl1,
                                                     rotbf2_fit_ampl2,rotbf2_fit_print_report,
                                                     rotbf2_fit_smoothing)

'''                                                     
#ax.legend()
#plt.show()

'''
fig,ax = plt.subplots()
levels = np.linspace(-0.0005,0.01,100)

cs = ax.contourf(rvs[:,i],dates,smoothed,levels, cmap='RdGy')

fig.colorbar(cs)
#cbar.ax.set_ylabel('verbosity coefficient')
ax.set_xlabel('RV [km/s]')
ax.set_ylabel(f'Time [days - {np.round(mjd_zero)}mjd]')
ax.set_title(f'River plot of ID: {ID}')
plt.show()
'''
def plot_river(ID,overlay=False):

    #Finding the vsini:
    bf_fit_path = f'/home/lakeclean/Documents/speciale/rv_data/rv_{ID}.txt'
    df = pd.read_csv(bf_fit_path)
    vsini1s = df['vsini1'].to_numpy()
    vsini2s = df['vsini2'].to_numpy()
    vrad1s = df['rv1'].to_numpy()
    vrad2s = df['rv2'].to_numpy()

    #correcting for vbary:
    def give_bary(epoch_date):
        bary_path = f'/home/lakeclean/Documents/speciale/order_file_log.txt'
        lines= open(bary_path).read().split('\n')
        for line in lines[:-1]:
            line = line.split(',')
            if line[1].strip() == 'science':
                if line[3].strip() == epoch_date: #Skipping duplicates
                    return float(line[-1].strip())/1000


    
    date_len = len(path + ID)+1
    folder_dates = glob.glob(path + ID + '/*')

    def sorter(x):
        return Time(x[date_len:]).mjd

    folder_dates = sorted(folder_dates,key=sorter)


    num_dates = len(folder_dates)
    dates = []
    bfs = np.zeros(shape=(401,num_dates))
    smoothed = np.zeros(shape=(num_dates,401))
    rvs = np.zeros(shape=(401,num_dates))
    v_barys = np.zeros(num_dates)

    order_sum = 40 #the number of orders that is summed
    for i,folder_date in enumerate(folder_dates):
        dates.append(Time(folder_date[date_len:]).mjd)
        v_barys[i] = give_bary(folder_date[date_len:])
        
        for j in range(order_sum):
            try:
                df = pd.read_csv(folder_date + f'/data/order_{j+20}_broadening_function.txt')
            except:
                print(folder_date+' could not be found. If 2024-07-13T00:26:25.672 then its a bad spec')
                continue
            
            df = df.to_numpy()
            bfs[:,i] = df[:,1]
            rvs[:,i] = df[:,0]
            smoothed[i,:] += df[:,2]

    smoothed = smoothed/order_sum

    mjd_zero = min(dates)

    dates = np.array(dates) - mjd_zero

    rv_region = 401
    fig,ax = plt.subplots()
    levels = np.linspace(-0.0005,0.07,100)
    #Shifting the broadening funcitons by the barycentric correction
    proxy_smoothed = np.zeros(shape=(num_dates,rv_region)) #The shifted wavelengths
    for i in range(num_dates):
        shift = int(v_barys[i])
        for j in np.arange(shift+40,rv_region-40,1):
            proxy_smoothed[i][j-shift] = smoothed[i][j]

    cs = ax.contourf(rvs[:,i],dates,proxy_smoothed,levels, cmap='RdGy')

    fig.colorbar(cs)
    #cbar.ax.set_ylabel('verbosity coefficient')
    ax.set_xlabel('RV [km/s]')
    ax.set_ylabel(f'Time [days - {np.round(mjd_zero)}mjd]')
    ax.set_title(f'River plot of ID: {ID}')
    ax.set_xlim(-100,50)
    plt.show()

    
#plot_river('KIC-10454113')


























