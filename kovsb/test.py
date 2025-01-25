# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:13:09 2024

@author: Jonatan Rudrasingam
"""

import sys

module_dir = "C:\\Users\\Jonatan\\Documents\\Jonatan\\Speciale"

sys.path.append(module_dir)

import numpy as np
import matplotlib.pyplot as plt
import orbit
import sboppy_jr as sboppy

plt.close("all")

time_rva, rva, rva_err, instrument = np.loadtxt("C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\rv_kic9025370_a.txt", skiprows = 1).T
time_rvb, rvb, rvb_err, instrument = np.loadtxt("C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\rv_kic9025370_b.txt", skiprows = 1).T

plt.figure()
plt.plot(time_rva, rva, 'r.')
plt.plot(time_rvb, rvb, 'b.')
plt.show()

period = 196.367073
phase_a = np.remainder(time_rva, period)/period
phase_b = np.remainder(time_rvb, period)/period

sys_a = -10

w = 360

ecc = 0.29602530

time_rva_model = np.linspace(np.min(time_rva), np.max(time_rva), 2000)
rva_model = orbit.radial_velocity(time_rva_model, 17, ecc, 117, period, 2460388, sys_a)
phase_a_long = np.remainder(time_rva_model, period)/period
sort_a = np.argsort(phase_a_long)


time_rvb_model = np.linspace(np.min(time_rvb), np.max(time_rvb), 2000)
rvb_model = orbit.radial_velocity(time_rvb_model, 24, ecc, 117 + 180, period, 2460388, sys_a)
phase_b_long = np.remainder(time_rvb_model, period)/period
sort_b = np.argsort(phase_b_long)

plt.figure()
plt.plot()
plt.plot(phase_a_long[sort_a], rva_model[sort_a], 'k')
plt.plot(phase_a, rva, 'r.')
plt.plot(phase_b_long[sort_b], rvb_model[sort_b], 'y')
plt.plot(phase_b, rvb, 'b.')
plt.show()

rva_err = np.zeros(len(time_rva)) + 1 # np.abs(result_r1 - result_w1)
rvb_err = np.zeros(len(time_rvb)) + 1 # np.abs(result_r2 - result_w2)
rv_data = np.transpose(np.array([rva, rvb, rva_err, rvb_err]))
sys.exit()
res = sboppy.plot_radvel(t = time_rva, data = rv_data, k = [15, 23], e = ecc, w = w, 
                         period = period, v0 = [32, 32], i = 113)