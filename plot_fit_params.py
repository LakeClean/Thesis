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
all_IDs, all_dates, all_vhelios = [], [], []
for line in lines[:-1]:
    line = line.split(',')
    if line[1].strip() == 'science':
        if line[0].strip() not in all_IDs:
            all_IDs.append(line[0].strip())
        all_dates.append(line[3].strip())
        all_vhelios.append(float(line[5].strip()))
    

#ID = 'KIC-12317678'#'KIC-9025370'
for ID in ['KIC9652971']:#all_IDs: #['KIC-12317678']:
    print(ID)
    path = f'/home/lakeclean/Documents/speciale/target_analysis/'
    correction = len(path + ID) +1
    files = glob.glob(path + ID + '/*')

    rvs = np.zeros(shape= (len(files),4))
    mjds = np.zeros(len(files))
    for i, file in enumerate(files):
        date = file[correction:]
        mjds[i] = Time(date).mjd
        for all_date, all_vhelio in zip(all_dates,all_vhelios):
            if all_date == date:
                v_helio = all_vhelio


        vrad1 = []
        vrad2= []
        ampl1= []
        ampl2= []
        vsini1= []
        vsini2= []
        gwidth= []
        limbd= []
        const= []
        try:
            path =  f'{file}/data/bf_fit_params.txt'
            lines = open(path).read().split('\n')
        except:
            print('{file}/data/bf_fit_params.txt could not be found')
            continue
        

        for line in lines[1:-1]:
            line  =line.split(',')
            ampl1.append(float(line[2]))
            ampl2.append(float(line[3]))
            #We choose the rad1 as the one with the highest amplitude
            if float(line[2]) < float(line[3]):
                vrad1.append(float(line[0]) + v_helio)
                vrad2.append(float(line[1]) + v_helio)
                vsini1.append(float(line[4]))
                vsini2.append(float(line[5]))
                gwidth.append(float(line[6]))
                limbd.append(float(line[7]))
                const.append(float(line[8]))
            else:
                vrad1.append(float(line[1]) + v_helio)
                vrad2.append(float(line[0]) + v_helio)
                vsini1.append(float(line[5]))
                vsini2.append(float(line[4]))
                gwidth.append(float(line[6]))
                limbd.append(float(line[7]))
                const.append(float(line[8]))
                

        
        fig, ax = plt.subplots()
        limits = [20,60]
        ax.scatter(range(len(vrad1)), vrad1,label=f'{np.median(vrad1)}')
        ax.scatter(range(len(vrad2)), vrad2,label=f'{np.std(vrad1)}')


        fit = fitting.LinearLSQFitter()
        line_init = models.Linear1D()
        fitted_line = fit(line_init, range(len(vrad1[limits[0]:limits[1]])), vrad1[limits[0]:limits[1]])
        slope = fitted_line.slope.value
        intercept = fitted_line.intercept.value
        ax.plot(np.linspace(limits[0],limits[1],len(vrad1[limits[0]:limits[1]])),fitted_line(range(len(vrad1[limits[0]:limits[1]]))))
        
        ax.legend()
        
        #plt.show()
        plt.close()
        
        rvs[i,0] = np.mean(vrad1[limits[0]:limits[1]])
        rvs[i,1] = np.mean(vrad2[limits[0]:limits[1]])
        rvs[i,2] = np.std(vrad1[limits[0]:limits[1]])/np.sqrt(len(vrad1[limits[0]:limits[1]]))
        rvs[i,3] = np.std(vrad2[limits[0]:limits[1]])/np.sqrt(len(vrad1[limits[0]:limits[1]]))
        
        


    fig, ax  = plt.subplots()



    ax.scatter(mjds,rvs[:,0],color='r')
    #ax.errorbar(mjds,rvs[:,0],rvs[:,2],fmt='o',capsize=2,color='r')
    ax.scatter(mjds,rvs[:,1],color='b')
    #ax.errorbar(mjds,rvs[:,1],rvs[:,3],fmt='o',capsize=2,color='b')

    #ax.scatter(np.array(mjds)+103,rv1s,color='g')
    #ax.errorbar(mjds,rvs[:,0],rvs[:,2],fmt='o',capsize=2,color='r')
    #ax.scatter(np.array(mjds)+103,rv2s,color='cyan')
    #ax.errorbar(mjds,rvs[:,1],rvs[:,3],fmt='o',capsize=2,color='b')



#Sorting points:

def sorter1(x):
    index = np.where(x == rvs[:,0])[0]
    return mjds[index]
def sorter2(x):
    index = np.where(x == rvs[:,1])[0]
    return mjds[index]

rv1s = sorted(rvs[:,0],key=sorter1)
rv2s = sorted(rvs[:,1],key=sorter2)
mjds = sorted(mjds)



    

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
K_guess, medialline_guess = guess_params(mjds,rv1s,100)
period_guess = 100

#KIC-12317678
#fit = sb.fit_radvel_SB1(mjds,rv1s,k=K_guess,e=0.50,
#                        w=100,p=period_guess,v0=medialline_guess[0])

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

proxy_time = np.linspace(min(mjds),max(mjds),1000)
fit_rvs = sb.radial_velocity(proxy_time,k=k,e=e,w=w,p=p,t0=t0,v0=v0)
ax.plot(proxy_time,fit_rvs)

plt.show()
'''

##################### SB2 ######################

K1_guess, medialline1_guess = guess_params(mjds,rv1s,100)
K2_guess, medialline2_guess = guess_params(mjds,rv2s,100)
period_guess = 100

print(K1_guess,K2_guess)
print(medialline1_guess)


rvs = np.zeros(shape=(len(rv1s),2))
rvs[:,0] = np.array(rv1s)
rvs[:,1] = np.array(rv2s)
fit = sb.fit_radvel_SB2(mjds,rvs,k=[K1_guess,K2_guess],e=0.4,
                        w=250,p=period_guess,v0=medialline1_guess[0])

k1 = fit.params['k1'].value
k2 = fit.params['k2'].value
e = fit.params['e'].value
w = fit.params['w'].value
p = fit.params['p'].value
t0 = fit.params['t0'].value
v0 = fit.params['v0_1'].value

print(fit.params)

proxy_time = np.linspace(min(mjds),max(mjds),1000)

fit_rvs = sb.radial_velocity(proxy_time,k=k1,e=e,w=w,p=p,t0=t0,v0=v0)
#ax.plot(proxy_time,-fit_rvs)

fit_rvs = sb.radial_velocity(proxy_time,k=k2,e=e,w=w,p=p,t0=t0,v0=v0)
#ax.plot(proxy_time,fit_rvs)

plt.show()







'''
def findOmega(new_v1s, BJD,fine=0.00001):
    #Interpolating and finding omega1
    steps= np.linspace(0,max(BJD),10000)

    interp = np.interp(steps,BJD,new_v1s)

    medialline = (max(new_v1s)+min(new_v1s))/2
    

    dist=np.zeros(len(steps))
    for i in range(len(steps)):
        dist[i] = interp[i] - medialline

        
    
    steps_above = steps[np.where(dist >0)[0]]
    steps_below = steps[np.where(dist <0)[0]]
    interp_above = interp[np.where(dist >0)[0]]
    interp_below = interp[np.where(dist <0)[0]]
    above = dist[np.where(dist >0)[0]]
    below = abs(dist[np.where(dist<0)[0]])
    
    half_period = []
    above_x = []
    above_y = []
    below_x = []
    below_y = []
    
    for i in range(len(below)):
        for j in range(len(above)):
            if above[j] < below[i]+fine : 
                if above[j] > below[i]-fine :
                    #print(above[j], below[i])
                    index_above = np.where(above[j]==above)[0]
                    index_below = np.where(below[i]==below)[0]
                    #print('indexs of above and below: ', np.where(above[j]==above)[0],np.where(below[i]==below)[0])
                    #print(interp_above[index_above],interp_below[index_below])
                    #print(steps_above[index_above],steps_below[index_below])
                    
                    ax.scatter(steps_above[index_above],interp_above[index_above])
                    ax.scatter(steps_below[index_below],interp_below[index_below])
                    #print(steps_above[index_above]-steps_below[index_below])
                    #print('K',max(interp),min(interp),( max(interp)-min(interp))/2)
                    
                    above_x = np.append(above_x,steps_above[index_above])
                    above_y = np.append(above_y,interp_above[index_above])
                    
                    below_x = np.append(below_x,steps_below[index_below])
                    below_y = np.append(below_y,interp_below[index_below])
                    half_period = np.append(half_period,steps_above[index_above]-steps_below[index_below])
    print(half_period)
    ax.hlines(medialline,-10,BJD[-1]+10,color='black')
    ax.vlines(above_x[-1],medialline,above_y[-1],label=above_y[-1]-medialline)
    ax.vlines(below_x[-1],below_y[-1],medialline, label=below_y[-1]-medialline)


findOmega(rvs[:,0],mjds)  
'''
