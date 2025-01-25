# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:12:24 2024

@author: Jonatan Rudrasingam
"""

import numpy as np

######################################################################################################################
# Input files
######################################################################################################################
# Radial velocity
rv_A_dir = "data/rv_kic9025370_a.txt"
rv_B_dir = "data/rv_kic9025370_b.txt"

# Relative astrometry file
rel_ast_dir = "data/rel_ast_chi_dra.txt"

# Gaia Thiele Innes
gaia_thiele_innes = "data/kic9025370_gti.txt"
C_matrix = "data/KIC4914923_gaia_cmat.npy"

######################################################################################################################
# Parameters and priors
######################################################################################################################
# (Value of initial guess, [lower limit, upper limit] of uniform prior)
t0 = (1020, [-np.inf, np.inf]) #226.20482746
k_a = (17.8, [0, np.inf]) #17.36107385
k_b = (17.0790718, [0, np.inf]) #24.26864186
a = (0.1244, [0, np.inf])
# Photocentric semi-major axis in AU
a0 = (1.298421318237935, [0, np.inf]) #0
# Eccentricity
e = (0.28983278, [0, 1]) #0.42334463
# Inclination
i = (113.90690082389432, [0, 180]) #74
w = (349.331402, [0, 360]) #117.98434577
W = (177.71985941863326, [0, 360]) #230.35
p = (196.42, [10, 390]) #(280.57075453, [270, 290])
# System velocity in km/s
v0_a = (-14, [-20, -10])
# System velocity in km/s (B)
v0_b = (-10, [-20, -5])
# Difference between system velocity in km/s
dvb = (4, [-np.inf, np.inf])
# Parallax in mas
pi = (8.09472464236166, [-np.inf, np.inf]) 

######################################################################################################################
# MCMC options
######################################################################################################################
nwalkers = 64
iterations = 10000
burnin = 5000
thin = 15
chain_size_plot = 0 #200

######################################################################################################################
# Plotting options
######################################################################################################################
show_err = False

######################################################################################################################
# Save
######################################################################################################################
params_sb1 = np.array([k_a[0], t0[0], e[0], w[0], p[0], v0_a[0]])
params_sb2 = np.array([k_a[0], k_b[0], t0[0], e[0], w[0], p[0], v0_a[0], v0_b[0]])
params_ast = np.array([t0[0], a[0], e[0], i[0], w[0], W[0], p[0]])
params_gaia_ti = np.array([t0[0], a0[0], e[0], i[0], w[0], W[0], p[0], pi[0]])
params_comb1 = np.array([k_a[0], t0[0], a[0], e[0], i[0], w[0], W[0], p[0], v0_a[0]])
params_comb2 = np.array([k_a[0], k_b[0], t0[0], a[0], e[0], i[0], w[0], W[0], p[0], v0_a[0], v0_b[0]])
params_comb3 = np.array([k_a[0], t0[0], a0[0], e[0], i[0], w[0], W[0], p[0], v0_a[0], pi[0]])
params_comb4 = np.array([k_a[0], k_b[0], t0[0], a0[0], e[0], i[0], w[0], W[0], p[0], v0_a[0], v0_b[0], pi[0]])

priors_sb1 = np.array([k_a[1], t0[1], e[1], w[1], p[1], v0_a[1]])
priors_sb2 = np.array([k_a[1], k_b[1], t0[1], e[1], w[1], p[1], v0_a[1], v0_b[1]])
priors_sb2_m = np.array([k_a[1], k_b[1], t0[1], e[1], w[1], p[1], v0_a[1], dvb[1]])
priors_ast = np.array([t0[1], a[1], e[1], i[1], w[1], W[1], p[1]])
priors_gaia_ti = np.array([t0[1], a0[1], e[1], i[1], w[1], W[1], p[1], pi[1]])
priors_comb1 = np.array([k_a[1], t0[1], a[1], e[1], i[1], w[1], W[1], p[1], v0_a[1]])
priors_comb2 = np.array([k_a[1], k_b[1], t0[1], a[1], e[1], i[1], w[1], W[1], p[1], v0_a[1], v0_b[1]])
priors_comb2_m = np.array([k_a[1], k_b[1], t0[1], a[1], e[1], i[1], w[1], W[1], p[1], v0_a[1], dvb[1]])
priors_comb3 = np.array([k_a[1], t0[1], a0[1], e[1], i[1], w[1], W[1], p[1], v0_a[1], pi[1]])
priors_comb4 = np.array([k_a[1], k_b[1], t0[1], a0[1], e[1], i[1], w[1], W[1], p[1], v0_a[1], v0_b[1], pi[1]])
priors_comb4_m = np.array([k_a[1], k_b[1], t0[1], a0[1], e[1], i[1], w[1], W[1], p[1], v0_a[1], dvb[1], pi[1]])

labels_sb1 = [r"$k_A$", r"$t_0$", r"$e$", r"$\omega\,$", r"$p$", r"$\gamma$"]
labels_sb1_ = ["kA", "t0", "e", "w", "p", "v0"]
labels_sb2 = [r"$k_A$", r"$k_B$", r"$t_0$", r"$e$", r"$\omega\,$", r"$p$", r"$\gamma\,_{A}$", r"$\gamma\,_{B}$"]
labels_sb2_ = ["kA", "kB", "t0", "e", "w", "p", "v0_A", "v0_B"]
labels_ast = [r"$t_0$", r"$a$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$"]
labels_ast_ = ["t0", "a", "e", "i", "w", "W", "p"]
labels_gaia_ti  = [r"$t_0$", r"$a_0$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$", r"$\pi\,$"]
labels_gaia_ti_ = ["t0", "a0", "e", "i", "w", "W", "p", "pi"]
labels_comb1 = [r"$k_A$", r"$t_0$", r"$a$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$", r"$\gamma\,_{A}$"]
labels_comb1_ = np.array(["kA", "t0", "a", "e", "i", "w", "W", "p", "v0_A"])
labels_comb2 = [r"$k_A$", r"$k_B$", r"$t_0$", r"$a$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$", r"$\gamma\,_{A}$", r"$\gamma\,_{B}$"]
labels_comb2_ = np.array(["kA", "kB", "t0", "a", "e", "i", "w", "W", "p", "v0_A", "v0_B"])
labels_comb3 = [r"$k_A$", r"$t_0$", r"$a_0$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$", r"$\gamma\,_{A}$", r"$\pi\,$"]
labels_comb3_ = np.array(["kA", "t0", "a0", "e", "i", "w", "W", "p", "v0_A", "pi"])
labels_comb4 = [r"$k_A$", r"$k_B$", r"$t_0$", r"$a_0$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$", r"$\gamma\,_{A}$", r"$\gamma\,_{B}$", r"$\pi\,$"]
labels_comb4_ = np.array(["kA", "kB", "t0", "a0", "e", "i", "w", "W", "p", "v0_A", "v0_B", "pi"])
