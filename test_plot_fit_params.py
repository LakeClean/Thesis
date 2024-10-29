import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd
from astropy.modeling import models, fitting
import sboppy as sb
from astropy.time import Time

path = f'/home/lakeclean/Documents/speciale/order_file_log.txt'

lines= open(path).read().split('\n')
all_IDs, all_dates, all_vhelios,files= [], [], [], []
for line in lines[:-1]:
    line = line.split(',')
    if line[1].strip() == 'science':
        all_IDs.append(line[0].strip())
        all_dates.append(line[3].strip())
        all_vhelios.append(float(line[5].strip()))
        files.append(line[2].strip())
    
def fit_line(x,y):
    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()
    fitted_line = fit(line_init, x, y)
    slope = fitted_line.slope.value
    intercept = fitted_line.intercept.value
    new_x = np.linspace(min(x),max(x),10)
    return x,fitted_line(x),slope
#ID = 'KIC-12317678'#'KIC-9025370' KIC-9693187 KIC9652971

rvs = np.zeros(shape= (len(files),4))
mjds = np.zeros(len(files))
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

#Going through each file
for i,date in enumerate(all_dates):  # in ['KIC-9693187']:#all_IDs: #['KIC-12317678']:
    path = f'/home/lakeclean/Documents/speciale/target_analysis/'+ f'{all_IDs[i]}/{date}/data/bf_fit_params.txt'
    date = all_dates[i]
    mjds[i] = Time(date).mjd
    v_helio = all_vhelios[i]

    try:
        lines = open(path).read().split('\n')
    except:
        print(f'{path} could not be found')
        continue

    #Going through each order
    for j,line in enumerate(lines[1:-1]):
        line  =line.split(',')
        ampl1s[i,j] = float(line[2])
        ampl2s[i,j] = float(line[3])
        #We choose the rad1 as the one with the highest amplitude
        if float(line[2]) < float(line[3]):
            vrad1s[i,j] =float(line[0]) + v_helio
            vrad2s[i,j] = float(line[1]) + v_helio
            vsini1s[i,j] = float(line[4])
            vsini2s[i,j] = float(line[5])
            gwidths[i,j] =float(line[6])
            limbds[i,j] =float(line[7])
            consts[i,j] =float(line[8])
        else:
            vrad1s[i,j] =float(line[1]) + v_helio
            vrad2s[i,j] = float(line[0]) + v_helio
            vsini1s[i,j] = float(line[5])
            vsini2s[i,j] = float(line[4])
            gwidths[i,j] =float(line[6])
            limbds[i,j] =float(line[7])
            consts[i,j] =float(line[8])
    

                
    limits = [20,60] #The region that looks best
    #Finding the offset from the median of the best region
    #offset1s[i,:] = np.median(vrad1s[i][limits[0]:limits[1]]) - vrad1s[i]
    #offset2s[i,:] = np.median(vrad2s[i][limits[0]:limits[1]]) - vrad2s[i]
    offset1s[i,:] = np.median(vrad1s[i]) - vrad1s[i]
    offset2s[i,:] = np.median(vrad2s[i]) - vrad2s[i]
        

#The offset for each order based on every file  
offset1s_median = np.median(offset1s,axis=0)
offset2s_median = np.median(offset2s,axis=0)


#Plotting the corrected and uncorrected rvs as funciton of order for each file

for i in range(len(files)):
    if not 'KIC9652971' == all_IDs[i]:
        continue
    fig, ax  = plt.subplots()

    #Correcting vrads for NOT wavelength issue
    corr_vrad1s = vrad1s[i]+offset1s_median
    corr_vrad2s = vrad2s[i]+offset2s_median

    #Selecting the best region of orders
    corr_vrad1s = corr_vrad1s[limits[0]:limits[1]]
    corr_vrad2s = corr_vrad2s[limits[0]:limits[1]]

    
    #Finding mean of vrads:
    good_vrad1s = []
    good_vrad2s = []
    for y1,y2 in zip(corr_vrad1s,corr_vrad2s):
        
        outlier_limit = 5
        temp_median1 = np.median(corr_vrad1s)
        temp_median2 = np.median(corr_vrad2s)

        if abs(y1 - temp_median1) < outlier_limit:
            good_vrad1s.append(y1)

        if abs(y2 - temp_median2) < outlier_limit:
            good_vrad2s.append(y2)

    ax.scatter(range(len(vrad1s[i])),vrad2s[i],label=f'{np.std(vrad2s[i][20:60])}')
    ax.scatter(range(len(vrad1s[i])),vrad2s[i]+offset1s_median, label=f'{np.std(good_vrad1s)}')

    x,y,slope = fit_line(range(len(good_vrad2s)),good_vrad2s)
    ax.plot(x,y,label=f'slope={slope}')

    x,y,slope = fit_line(range(len(good_vrad2s)),good_vrad2s)
    ax.plot(x,y,label=f'slope={slope}')

    ax.set_xlabel('Order')
    ax.set_ylabel('RV')
    ax.set_title(f'{all_IDs[i]}, {all_dates[i]}, vhelio={all_vhelios[i]}')
    ax.legend()

    plt.show()


ID = 'KIC9652971'
epoch_rv1s = []
epoch_rv2s = []
epoch_rv1_errs = []
epoch_rv2_errs = []
epoch_mjds = []

for i,all_ID in enumerate(all_IDs):
    if not ID == all_ID:
        continue

    #Correcting vrads for NOT wavelength issue
    corr_vrad1s = vrad1s[i]+offset1s_median
    corr_vrad2s = vrad2s[i]+offset2s_median

    #Selecting the best region of orders
    corr_vrad1s = corr_vrad1s[limits[0]:limits[1]]
    corr_vrad2s = corr_vrad2s[limits[0]:limits[1]]

    
    #Finding mean of vrads:
    good_vrad1s = []
    good_vrad2s = []
    for y1,y2 in zip(corr_vrad1s,corr_vrad2s):
        
        outlier_limit = 5
        temp_median1 = np.median(corr_vrad1s)
        temp_median2 = np.median(corr_vrad2s)

        if abs(y1 - temp_median1) < outlier_limit:
            good_vrad1s.append(y1)

        if abs(y2 - temp_median2) < outlier_limit:
            good_vrad2s.append(y2)

    
    epoch_rv1s.append(np.mean(good_vrad1s))
    epoch_rv2s.append(np.mean(good_vrad2s))

    epoch_rv1_errs.append(np.std(good_vrad1s)/np.sqrt(len(good_vrad1s)))
    epoch_rv2_errs.append(np.std(good_vrad2s)/np.sqrt(len(good_vrad2s)))
    epoch_mjds.append(mjds[i])
    
fig,ax = plt.subplots()
ax.set_title(f'ID: {ID}')
ax.errorbar(epoch_mjds, epoch_rv1s, epoch_rv1_errs,
           fmt='o',capsize=2,color='r')

ax.errorbar(epoch_mjds, epoch_rv2s, epoch_rv2_errs,
           fmt='o',capsize=2,color='b')

#ax.scatter(np.array(epoch_mjds)%99,epoch_rv1s)

plt.show()

    
        


def guess_params(ts, rv, period_guess):
    ################  Guessing parameters:  #######################

    #First we interpolate

    steps= np.linspace(min(ts),max(ts),100)

    interp = np.interp(steps,ts,rv)

    #ax.plot(steps,interp)
    #period based on eye estimate
    period_guess = period_guess

    #Finding medialline
    mediallines = np.linspace(min(interp),max(interp),10)
    area_diff = np.zeros(len(mediallines))
    for j,medialline in enumerate(mediallines):
        above_rvs = interp[np.where(interp>medialline)[0]]
        below_rvs = interp[np.where(interp<medialline)[0]]
        above_steps = steps[np.where(interp>medialline)[0]]
        below_steps = steps[np.where(interp<medialline)[0]]
        
        #Find area points that above and below:
        area_above = 0
        for i in range(len(above_steps)-1):
            x = above_steps
            y = above_rvs
            area_above += abs(x[i+1] - x[i]) * abs(y[i+1] - medialline)
                                                   
        area_below = 0
        for i in range(len(below_steps)-1):
            x = below_steps
            y = below_rvs
            area_below += abs(x[i+1] - x[i]) * abs(y[i+1] - medialline)
                                                
        area_diff[j] = abs(area_above -area_below)

        

    medialline_guess = mediallines[np.where(area_diff == min(area_diff))[0]]

    # Guessing K:
    K_guess = (max(interp) - min(interp))/2

    return K_guess, medialline_guess



################### SB1 #######################
'''
K_guess, medialline_guess = guess_params(epoch_mjds,epoch_rv1s,100)
print(K_guess)
print(medialline_guess)
period_guess = 99

fit = sb.fit_radvel_SB1(epoch_mjds,epoch_rv1s,k=K_guess,e=0.50,
                        w=50,p=period_guess,v0=medialline_guess[0])




#KIC-4914923
#fit = sb.fit_radvel_SB1(mjds,rv1s,k=K_guess,e=0.1,
#                        w=10,p=period_guess,v0=medialline_guess[0])
k = fit.params['k1'].value
e = fit.params['e'].value
w = fit.params['w'].value
p = fit.params['p'].value
t0 = fit.params['t0'].value
v0 = fit.params['v0_1'].value

print(fit.params)

proxy_time = np.linspace(min(epoch_mjds),max(epoch_mjds),1000)
fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
ax.plot(proxy_time,fit_rvs)


plt.show()
'''

##################### SB2 ######################

#epoch_mjds = epoch_mjds[1:]
#epoch_rv1s[0],epoch_rv2s[0] = epoch_rv2s[0],epoch_rv1s[0]

K1_guess, medialline1_guess = guess_params(epoch_mjds,epoch_rv1s,100)
K2_guess, medialline2_guess = guess_params(epoch_mjds,epoch_rv2s,100)
period_guess = 200

print(K1_guess,K2_guess)
print(medialline1_guess)


rvs = np.zeros(shape=(len(epoch_rv1s),2))
rvs[:,0] = np.array(epoch_rv1s)
rvs[:,1] = np.array(epoch_rv2s)
fit = sb.fit_radvel_SB2(epoch_mjds,rvs,k=[K1_guess,K2_guess],e=0.2,
                        w=200,p=period_guess,v0=medialline1_guess[0])

k1 = fit.params['k1'].value
k2 = fit.params['k2'].value
e = fit.params['e'].value
w = fit.params['w'].value
p = fit.params['p'].value
t0 = fit.params['t0'].value
v0 = fit.params['v0_1'].value

print(fit.params)

proxy_time = np.linspace(min(epoch_mjds),max(epoch_mjds),1000)

fit_rvs = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)
ax.plot(proxy_time,fit_rvs)

fit_rvs = sb.radial_velocity(proxy_time,k=k2,e=e,w=w,p=p,t0=t0,v0=-v0)
ax.plot(proxy_time,-fit_rvs)

plt.show()



#Really good guesses:
'''
#KIC-12317678
Parameters([('k1', <Parameter 'k1', value=18.146551986461894, bounds=[0.0:inf]>), ('e', <Parameter 'e', value=0.30842419358606826, bounds=[0.0:1.0]>),
('w', <Parameter 'w', value=278.733967684972, bounds=[0.0:360.0]>), ('p', <Parameter 'p', value=80.0608252327795, bounds=[-inf:inf]>),
('t0', <Parameter 't0', value=-68.1503729095733, bounds=[-inf:inf]>), ('v0_1', <Parameter 'v0_1', value=-41.08561830926434, bounds=[-inf:inf]>)])

#KIC-9693187
Parameters([('k1', <Parameter 'k1', value=29.189674847826193, bounds=[0.0:inf]>), ('k2', <Parameter 'k2', value=26.29127616755267, bounds=[0.0:inf]>),
('e', <Parameter 'e', value=0.4493994177095235, bounds=[0.0:1.0]>), ('w', <Parameter 'w', value=44.437936702501176, bounds=[0.0:360.0]>),
('p', <Parameter 'p', value=104.10440279140843, bounds=[-inf:inf]>), ('t0', <Parameter 't0', value=-258.5980636542009, bounds=[-inf:inf]>),
('v0_1', <Parameter 'v0_1', value=-9.622957650745287, bounds=[-inf:inf]>)])


#KIC-10454113
Looks like a really long period orbit. A straight line would fit well.


#KIC-4914923
Parameters([('k1', <Parameter 'k1', value=15.451681911047931, bounds=[0.0:inf]>), ('e', <Parameter 'e', value=0.20733061037149964, bounds=[0.0:1.0]>),
('w', <Parameter 'w', value=105.11050703896773, bounds=[0.0:360.0]>), ('p', <Parameter 'p', value=99.20286889338607, bounds=[-inf:inf]>),
('t0', <Parameter 't0', value=-166.2733050942353, bounds=[-inf:inf]>), ('v0_1', <Parameter 'v0_1', value=-24.398265475664452, bounds=[-inf:inf]>)])


#KIC4457331
All of the spectra have almost duplicates so actually not that many observations.

#EPIC-246696804
Too few points to fit

#EPIC-212617037
Too few points to fit

#EPIC-249570007
Too few points to fit

#KIC-9025370 Doesn't fit very well. Issue with the first point
Parameters([('k1', <Parameter 'k1', value=15.711641164783249 +/- 6.94, bounds=[0.0:inf]>), ('k2', <Parameter 'k2', value=16.480683912789605 +/- 7.25, bounds=[0.0:inf]>),
('e', <Parameter 'e', value=0.4710733560091602 +/- 0.23, bounds=[0.0:1.0]>), ('w', <Parameter 'w', value=0.001668171614703784 +/- 11.8, bounds=[0.0:360.0]>),
('p', <Parameter 'p', value=196.5017765473483 +/- 1.12, bounds=[-inf:inf]>), ('t0', <Parameter 't0', value=1044.634345374434 +/- 341, bounds=[-inf:inf]>),
('v0_1', <Parameter 'v0_1', value=-13.551532635800074 +/- 0.827, bounds=[-inf:inf]>)])


#EPIC-230748783
Too few points to fit


#EPIC-236224056
Too few points but does show some movemen't


#KIC4260884
Does vary but basically looks like a straight line. Period must be at least and quite likely much higher.

#KIC9652971
Looks like it is straight flat line.







'''

















      


