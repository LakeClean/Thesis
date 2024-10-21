import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd
from astropy.modeling import models, fitting


ID = 'KIC-4914923'
date = '2024-04-21T01:48:49.532'
path = f'/home/lakeclean/Documents/speciale/target_analysis/{ID}/{date}/data/'
path = path + 'bf_fit_params.txt.txt'


lines = open(path).read().split('\n')

vrad1 = []
vrad2= []
ampl1= []
ampl2= []
vsini1= []
vsini2= []
gwidth= []
limbd= []
const= []

for line in lines[1:-1]:
    line  =line.split(',')
    vrad1.append(float(line[0]))
    vrad2.append(float(line[1]))
    ampl1.append(float(line[2]))
    ampl2.append(float(line[3]))
    vsini1.append(float(line[4]))
    vsini2.append(float(line[5]))
    gwidth.append(float(line[6]))
    limbd.append(float(line[7]))
    const.append(float(line[8]))


fig, ax = plt.subplots()
ax.scatter(range(len(vrad1[20:80])), vrad1[20:80])


fit = fitting.LinearLSQFitter()
line_init = models.Linear1D()
fitted_line = fit(line_init, range(len(vrad1[20:80])), vrad1[20:80])
slope = fitted_line.slope.value
intercept = fitted_line.intercept.value
ax.plot(range(len(vrad1[20:80])),fitted_line(range(len(vrad1[20:80]))))
print(slope)

plt.show()

