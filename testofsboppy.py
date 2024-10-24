import sboppy as sb
import numpy as np
from random import random
import matplotlib.pyplot as plt
import shazam

#Generating data
ts = np.linspace(0,1000,100)

for i in range(len(ts)):
    num = random() * 10
    ts[i] = ts[i] + num

rvs = sb.radial_velocity(ts,k=11,e=0.42,w=24,p=213)

for i in range(len(ts)):
    num = random() * 10
    
    rvs[i] = rvs[i] + num
    
fig, ax = plt.subplots()

ax.scatter(ts,rvs)
#plt.show()


################  Guessing parameters:  #######################

#First we interpolate

steps= np.linspace(0,max(ts),10000)

interp = np.interp(steps,ts,rvs)

ax.plot(steps,interp)

#period based on eye estimate
period_guess = 200

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

# Guessing w:
steps= np.linspace(0,max(ts),10000)

interp = np.interp(steps,ts,rvs)
above_rvs = interp[np.where(interp>medialline_guess)[0]]
below_rvs = interp[np.where(interp<medialline_guess)[0]]
above_steps = steps[np.where(interp>medialline_guess)[0]]
below_steps = steps[np.where(interp<medialline_guess)[0]]

above_distance = abs(above_rvs - medialline_guess)
below_distance = abs(medialline_guess- below_rvs)

contender_i = []
contender_j = []
for i in range(len(above_distance)):
    for j in range(len(below_distance)):
        if abs(above_distance[i] - below_distance[j])<0.01:
            contender_i.append(i)
            contender_j.append(j)

    

print(len(contender_i))
best_contender_i = []
for i,j in zip(contender_i,contender_j):
    if abs(abs(above_steps[i] - below_steps[j])  - period_guess/2) <0.1:
        best_contender_i.append(i)


print(len(best_contender_i))
print(above_distance[best_contender_i])
print(np.degrees(np.arccos(np.radians(above_distance[best_contender_i])/K_guess)))
            
    

e_guess = ((max(interp) - medialline) /K_guess - 1) * 1/np.cos(1.5)

print(e_guess)


#Fitting
fit = sb.fit_radvel_SB1(ts,rvs,k=K_guess,e=0.01,w=300,p=200)
k = fit.params['k1'].value
e = fit.params['e'].value
w = fit.params['w'].value
p = fit.params['p'].value
t0 = fit.params['t0'].value

print(fit.params)
fit_rvs = sb.radial_velocity(ts,k=k,e=e,w=w,p=p,t0=t0)

ax.plot(ts,fit_rvs)


fit = sb.fit_radvel_SB1(ts,rvs,k=k,e=e,w=w,p=p,t0=t0)

k = fit.params['k1'].value
e = fit.params['e'].value
w = fit.params['w'].value
p = fit.params['p'].value
t0 = fit.params['t0'].value

print(fit.params)
fit_rvs = sb.radial_velocity(ts,k=k,e=e,w=w,p=p,t0=t0)

ax.plot(ts,fit_rvs,label='second fit')
ax.legend()
plt.show()



'''
        fit = fitting.LinearLSQFitter()
        line_init = models.Linear1D()
        fitted_line = fit(line_init, nwl, nfl)
        slope = fitted_line.slope.value
        slopes[i] = slope
        '''
