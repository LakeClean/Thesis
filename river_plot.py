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

order_sum = 40 #the number of orders that is summed
for i,folder_date in enumerate(folder_dates):
    dates.append(Time(folder_date[date_len:]).mjd)
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
    offset = dates[i]*0.01
    ax.plot(rvs[:,i],smoothed[i,:] + offset)

plt.show()

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
plt.show()

fig,ax = plt.subplots()
levels = np.linspace(-0.0005,0.01,100)

cs = ax.contourf(rvs[:,i],dates,smoothed,levels, cmap='RdGy')

fig.colorbar(cs)
#cbar.ax.set_ylabel('verbosity coefficient')
ax.set_xlabel('RV [km/s]')
ax.set_ylabel(f'Time [days - {np.round(mjd_zero)}mjd]')
ax.set_title(f'River plot of ID: {ID}')
plt.show()

    

    
    
