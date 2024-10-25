import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd
from astropy.modeling import models, fitting
import sboppy as sb
from astropy.time import Time

'''
Sricpt for constructing "river plot" of the movement of the stars

-We construct a list of the 30th order for all the times of the star
-We extract the bf for these
-we make a contour plot or river plot based on intensity of bf
'''

path = '/home/lakeclean/Documents/speciale/target_analysis/'
ID = 'KIC-4914923'


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
rvs = np.zeros(401)
for i,folder_date in enumerate(folder_dates):
    dates.append(Time(folder_date[date_len:]).mjd)
    df = pd.read_csv(folder_date + '/data/order_30_broadening_function.txt')
    df = df.to_numpy()
    bfs[:,i] = df[:,1]
    rvs = df[:,0]
    smoothed[i,:] = df[:,2]

mjd_zero = min(dates)
dates = np.array(dates) - mjd_zero

'''
fig, ax = plt.subplots()
offset = min(dates)
for i in range(len(dates)):
    offset = dates[i]*0.01
    ax.plot(rvs,smoothed[i,:] + offset)

plt.show()
'''

levels = np.linspace(-0.01,0.1,100)
plt.contourf(rvs,dates,smoothed,levels, cmap='RdGy')
plt.xlabel('RV [km/s]')
plt.ylabel(f'Time [days - {np.round(mjd_zero)}mjd]')
plt.title(f'River plot of ID: {ID}')
plt.show()

    

    
    
