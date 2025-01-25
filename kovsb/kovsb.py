# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:57:00 2023

@author: Jonatan Rudrasingam

Keplerian Orbit for Visual and Spectroscopic Binary (KOVSB)

"""
import sys

if sys.platform == "linux" or sys.platform == "linux2":
    module_dir = "/usr/users/au649504/py_modules"
elif sys.platform == "win32":
    module_dir = "C:\\Users\\Jonatan Rudrasingam\\Desktop\\Jonatan\\Modules"

sys.path.append(module_dir)
import numpy as np
import matplotlib.pyplot as plt
import emcee
import likelihoods
import plotter as kplt
import config
import corner
import arviz as az
import robust

plt.close("all")

######################################################################################################################
# MCMC options
######################################################################################################################
options = "SB1"
options = "SB2"
#options = "AST"
#options = "GAIA - Thiele Innes"
#options = "COMBINED (SB1 + AST)"
#options = "COMBINED (SB2 + AST)"
#options = "COMBINED (SB1 + GAIA TI)"
options = "COMBINED (SB2 + GAIA TI)"

# Multi instruments options
options = "SB1 (multi instruments)"
options = "SB2 (multi instruments)"
options = "COMBINED (SB1 (multi instruments) + AST)"
options = "COMBINED (SB2 (multi instruments) + AST)"
options = "COMBINED (SB1 (multi instruments) + GAIA TI)"
options = "COMBINED (SB2 (multi instruments) + GAIA TI)"

######################################################################################################################
# Setup
######################################################################################################################
if options == "SB1":
    time_rv, rva, rva_err, _ = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    data = (time_rv, rva, rva_err, config.priors_sb1)
    params = config.params_sb1
    labels = config.labels_sb1
    labels_ = config.labels_sb1_
    log_prior = likelihoods.log_prior_rv_sb1
    log_probability = likelihoods.log_probability_rv_sb1
    plot_res = kplt.plot_rv_sb1

if options == "SB2":
    time_rva, rva, rva_err, _ = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_rvb, rvb, rvb_err, _ = np.loadtxt(config.rv_B_dir, skiprows = 1).T
    data = (time_rva, time_rvb, rva, rvb, rva_err, rvb_err, config.priors_sb2)
    params = config.params_sb2
    labels = config.labels_sb2
    labels_ = config.labels_sb2_
    log_prior = likelihoods.log_prior_rv_sb2
    log_probability = likelihoods.log_probability_rv_sb2
    plot_res = kplt.plot_rv_sb2

if options == "AST":
    time_ast, rho, rho_err, theta, theta_err = np.loadtxt(config.rel_ast_dir, skiprows = 1).T
    # Calculate the correlation coefficient between theta and rho
    c = np.corrcoef((rho, theta))[0][1]
    data = (time_ast, rho, theta, rho_err, theta_err, c, config.priors_ast)
    params = config.params_ast
    params_ast = config.params_ast
    labels = config.labels_ast
    labels_ = config.labels_ast_
    log_prior = likelihoods.log_prior_ast
    log_probability = likelihoods.log_probability_ast
    plot_res = kplt.plot_ast

if options == "GAIA - Thiele Innes":
    params = config.params_gaia_ti
    gaia_ti = np.loadtxt(config.gaia_thiele_innes, skiprows = 1).T
    C_ma = np.load(config.C_matrix, allow_pickle = True)
    
    data = (gaia_ti, C_ma, config.priors_gaia_ti)
    labels = config.labels_gaia_ti
    labels_ = config.labels_gaia_ti_
    log_prior = likelihoods.log_prior_gti
    log_probability = likelihoods.log_probability_gti
    plot_res = kplt.plot_gti

if options == "COMBINED (SB1 + AST)":
    time_rva, rva, rva_err, _ = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_ast, rho, rho_err, theta, theta_err = np.loadtxt(config.rel_ast_dir, skiprows = 1).T
    # Calculate the correlation coefficient between theta and rho
    c = np.corrcoef((rho, theta))[0][1]
    
    data = (time_rva, rva, rva_err, time_ast, rho, theta, rho_err, theta_err, 
            c, config.priors_comb1)
    params = config.params_comb1
    labels = config.labels_comb1
    labels_ = config.labels_comb1_
    log_prior = likelihoods.log_prior_comb1
    log_probability = likelihoods.log_probability_comb1
    plot_res = kplt.plot_comb1

if options == "COMBINED (SB2 + AST)":
    time_rva, rva, rva_err, _ = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_rvb, rvb, rvb_err, _ = np.loadtxt(config.rv_B_dir, skiprows = 1).T
    time_ast, rho, rho_err, theta, theta_err = np.loadtxt(config.rel_ast_dir, skiprows = 1).T
    # Calculate the correlation coefficient between theta and rho
    c = np.corrcoef((rho, theta))[0][1]
    
    data = (time_rva, time_rvb, rva, rvb, rva_err, rvb_err, time_ast, 
            rho, theta, rho_err, theta_err, c, config.priors_comb2)
    params = config.params_comb2
    labels = config.labels_comb2
    labels_ = config.labels_comb2_
    log_prior = likelihoods.log_prior_comb2
    log_probability = likelihoods.log_probability_comb2
    plot_res = kplt.plot_comb2

if options == "COMBINED (SB1 + GAIA TI)":
    time_rva, rva, rva_err, _ = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    gaia_ti = np.loadtxt(config.gaia_thiele_innes, skiprows = 1).T
    C_ma = np.load(config.C_matrix, allow_pickle = True)
    
    data = (time_rva, rva, rva_err, gaia_ti, C_ma, config.priors_comb3)
    params = config.params_comb3
    labels = config.labels_comb3
    labels_ = config.labels_comb3_
    log_prior = likelihoods.log_prior_comb3
    log_probability = likelihoods.log_probability_comb3
    plot_res = kplt.plot_comb3

if options == "COMBINED (SB2 + GAIA TI)":
    time_rva, rva, rva_err, _ = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_rvb, rvb, rvb_err, _ = np.loadtxt(config.rv_B_dir, skiprows = 1).T
    gaia_ti = np.loadtxt(config.gaia_thiele_innes, skiprows = 1).T
    C_ma = np.load(config.C_matrix, allow_pickle = True)
    
    data = (time_rva, time_rvb, rva, rvb, rva_err, rvb_err, gaia_ti, C_ma, config.priors_comb4)
    params = config.params_comb4
    labels = config.labels_comb4
    labels_ = config.labels_comb4_
    log_prior = likelihoods.log_prior_comb4
    log_probability = likelihoods.log_probability_comb4
    plot_res = kplt.plot_comb4
    

# Multi instruments
if options == "SB1 (multi instruments)":
    time_rv, rva, rva_err, instrument = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    instrument = instrument.astype(int)
    unique_instrument = np.unique(instrument)
    len_instr = len(unique_instrument)
    
    data = (time_rv, rva, rva_err, instrument, config.priors_sb1)
    params = np.concatenate((config.params_sb1[0:5], np.repeat(config.params_sb1[-1], len_instr)))
    
    v0_labels = np.array([fr"$\gamma_{i}$" for i in np.arange(len_instr)])
    labels = np.concatenate((config.labels_sb1[0:5], v0_labels))
    v0_labels_ = np.array([f"v0_{i}" for i in np.arange(len_instr)])
    labels_ = np.concatenate((config.labels_sb1_[0:5], v0_labels_))
    
    log_prior = likelihoods.log_prior_rv_sb1_multi
    log_probability = likelihoods.log_probability_rv_sb1_multi
    plot_res = kplt.plot_rv_sb1_multi

if options == "SB2 (multi instruments)":
    time_rva, rva, rva_err, instrument_a = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_rvb, rvb, rvb_err, instrument_b = np.loadtxt(config.rv_B_dir, skiprows = 1).T
    instrument_a, instrument_b = instrument_a.astype(int), instrument_b.astype(int)
    unique_instrument = np.unique(np.concatenate((instrument_a, instrument_b)))
    len_instr = len(unique_instrument)
    
    data = (time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, config.priors_sb2_m)
    params = np.concatenate((config.params_sb2[0:6], np.repeat(config.params_sb2[6], len_instr), np.array([config.dvb[0]])))
    
    v0_labels = np.array([fr"$\gamma_{i}$" for i in np.arange(len_instr)])
    labels = np.concatenate((config.labels_sb2[0:6], v0_labels, np.array([r"$\Delta\,_{B}$"])))
    v0_labels_ = np.array([f"v0_{i}" for i in np.arange(len_instr)])
    labels_ = np.concatenate((config.labels_sb2_[0:6], v0_labels_, np.array(["Delta_B"])))
    
    log_prior = likelihoods.log_prior_rv_sb2_multi
    log_probability = likelihoods.log_probability_rv_sb2_multi
    plot_res = kplt.plot_rv_sb2_multi

if options == "COMBINED (SB1 (multi instruments) + AST)":
    time_rva, rva, rva_err, instrument_a = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    instrument_a = instrument_a.astype(int)
    unique_instrument = np.unique(instrument_a)
    len_instr = len(unique_instrument)
    
    time_ast, rho, rho_err, theta, theta_err = np.loadtxt(config.rel_ast_dir, skiprows = 1).T
    # Calculate the correlation coefficient between theta and rho
    c = np.corrcoef((rho, theta))[0][1]
    
    data = (time_rva, rva, rva_err, instrument_a, 
            time_ast, rho, theta, rho_err, theta_err, c, 
            config.priors_comb1)
    
    params = np.concatenate((config.params_comb1[0:8], np.repeat(config.params_comb1[8], len_instr)))

    v0_labels = np.array([fr"$\gamma_{i}$" for i in np.arange(len_instr)])
    labels = np.concatenate((config.labels_comb2[0:8], v0_labels))
    v0_labels_ = np.array([f"v0_{i}" for i in np.arange(len_instr)])
    labels_ = np.concatenate((config.labels_comb2_[0:8], v0_labels_))
    
    log_prior = likelihoods.log_prior_comb1_multi
    log_probability = likelihoods.log_probability_comb1_multi
    plot_res = kplt.plot_comb1_multi

if options == "COMBINED (SB2 (multi instruments) + AST)":
    time_rva, rva, rva_err, instrument_a = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_rvb, rvb, rvb_err, instrument_b = np.loadtxt(config.rv_B_dir, skiprows = 1).T 
    instrument_a, instrument_b = instrument_a.astype(int), instrument_b.astype(int)
    unique_instrument = np.unique(np.concatenate((instrument_a, instrument_b)))
    len_instr = len(unique_instrument)
    
    time_ast, rho, rho_err, theta, theta_err = np.loadtxt(config.rel_ast_dir, skiprows = 1).T
    # Calculate the correlation coefficient between theta and rho
    c = np.corrcoef((rho, theta))[0][1]
    
    data = (time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, 
            time_ast, rho, theta, rho_err, theta_err, c, config.priors_comb2)
    
    params = np.concatenate((config.params_comb2[0:9], np.repeat(config.params_comb2[9], len_instr), np.array([config.dvb[0]])))

    v0_labels = np.array([fr"$\gamma_{i}$" for i in np.arange(len_instr)])
    labels = np.concatenate((config.labels_comb2[0:9], v0_labels, np.array([r"$\Delta\,_{B}$"])))
    v0_labels_ = np.array([f"v0_{i}" for i in np.arange(len_instr)])
    labels_ = np.concatenate((config.labels_comb2_[0:9], v0_labels_, np.array(["Delta_B"])))
    
    log_prior = likelihoods.log_prior_comb2_multi
    log_probability = likelihoods.log_probability_comb2_multi
    plot_res = kplt.plot_comb2_multi

if options == "COMBINED (SB1 (multi instruments) + GAIA TI)":
    time_rva, rva, rva_err, instrument_a = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    instrument_a = instrument_a.astype(int)
    unique_instrument = np.unique(instrument_a)
    len_instr = len(unique_instrument)
    
    gaia_ti = np.loadtxt(config.gaia_thiele_innes, skiprows = 1).T
    C_ma = np.load(config.C_matrix, allow_pickle = True)
    
    data = (time_rva, rva, rva_err, instrument_a, gaia_ti, C_ma, config.priors_comb3)
    
    params = np.concatenate((config.params_comb3[0:8], np.repeat(config.params_comb3[8], len_instr), np.array([config.params_comb3[9]])))
    
    v0_labels = np.array([fr"$\gamma_{i}$" for i in np.arange(len_instr)])
    labels = np.concatenate((config.labels_comb3[0:8], v0_labels, np.array([config.labels_comb3[9]])))
    v0_labels_ = np.array([f"v0_{i}" for i in np.arange(len_instr)])
    labels_ = np.concatenate((config.labels_comb3_[0:8], v0_labels_, np.array([config.labels_comb3_[9]])))
    
    log_prior = likelihoods.log_prior_comb3_multi
    log_probability = likelihoods.log_probability_comb3_multi
    plot_res = kplt.plot_comb3_multi
    
if options == "COMBINED (SB2 (multi instruments) + GAIA TI)":
    time_rva, rva, rva_err, instrument_a = np.loadtxt(config.rv_A_dir, skiprows = 1).T
    time_rvb, rvb, rvb_err, instrument_b = np.loadtxt(config.rv_B_dir, skiprows = 1).T
    instrument_a, instrument_b = instrument_a.astype(int), instrument_b.astype(int)
    unique_instrument = np.unique(np.concatenate((instrument_a, instrument_b)))
    len_instr = len(unique_instrument)
    
    gaia_ti = np.loadtxt(config.gaia_thiele_innes, skiprows = 1).T
    C_ma = np.load(config.C_matrix, allow_pickle = True)
    
    data = (time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, gaia_ti, C_ma, config.priors_comb4_m)
    params = np.concatenate((config.params_comb4[0:9], np.repeat(config.params_comb4[9], len_instr), np.array([config.dvb[0], config.pi[0]])))
    
    v0_labels = np.array([fr"$\gamma_{i}$" for i in np.arange(len_instr)])
    labels = np.concatenate((config.labels_comb4[0:9], v0_labels, np.array([r"$\Delta\,_{B}$", config.labels_comb4_[11]])))
    v0_labels_ = np.array([f"v0_{i}" for i in np.arange(len_instr)])
    labels_ = np.concatenate((config.labels_comb4_[0:9], v0_labels_, np.array(["Delta_B", config.labels_comb4_[11]])))
    
    log_prior = likelihoods.log_prior_comb4_multi
    log_probability = likelihoods.log_probability_comb4_multi
    plot_res = kplt.plot_comb4_multi

combination_options = ("COMBINED (SB1 + AST)", "COMBINED (SB2 + AST)", 
                       "COMBINED (SB1 + GAIA TI)", "COMBINED (SB2 + GAIA TI)")

multi_instruments_options = ("SB1 (multi instruments)", 
                             "SB2 (multi instruments)", 
                             "COMBINED (SB1 (multi instruments) + AST)", 
                             "COMBINED (SB2 (multi instruments) + AST)", 
                             "COMBINED (SB1 (multi instruments) + GAIA TI)", 
                             "COMBINED (SB2 (multi instruments) + GAIA TI)")

######################################################################################################################
# MCMC
######################################################################################################################
def run_emcee(data: tuple, p0: np.ndarray, nwalkers: int, iterations: int, 
              log_probability: callable, log_prior: callable):
    """
    Runs MCMC on the input data with number of iterations and number as walkers
    as input. Takes both the log proabability function and the log prior function
    as input, so emcce can run on astrometric data, radial velocity data, or
    combined astrometric and radial velocity data.
    Returns an unflatted emcee samples, which have to be flatted (thines, and
    burnin removed)
    
    :params:
      data            : tuple, the astrometric and spectroscopic parameters
      p0              : array, the start guess for each parameters
      nwalkers        : int, numbers of walkers
      iterations      : int, number of MCMC iterations
      log_probability : callable, the log probability function
      log_prior       : callable, the log prior function
      
    :return:
      sampler        : ensemble.EnsembleSampler, the emcee sampler
      
    """
    
    # Create the start position of the walkers using the start guess
    ndim = len(p0)
    
    """ # Below doesn't work as inteded
    get_pos = True
    lp = np.zeros(ndim, dtype = "object")
    # While loop to check if walkers are outside priors
    while get_pos == True:
        pos = p0 + 1e-3*np.random.randn(nwalkers, ndim) #-4
        for i in pos:
            lp = np.append(lp, log_prior(i))
        if lp.any() != -np.inf:
            get_pos = False
    """
    pos = p0 + 1e-3*np.random.randn(nwalkers, ndim) #-4
    # Run the MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = data)
    sampler.run_mcmc(pos, iterations, progress = True)
    
    return sampler


def get_params(flat_samples: np.ndarray, std: float = 0.68):
    """
    Get the mean of the parameter using robust, and calculate 
    highest density interval (HDI) using ArviZ.
    
    :params:
        flat_samples : array, the flatten MCMC sample
        std          : float, highest density interval
        
    :return:
        mcmc        : array, contains the lower limit, mean, and the upper limit of (HDI)
        q           : array, the difference between mean and lower limit, and the mean and upper limit
        
    """
    
    ndim = len(flat_samples.T)
    mcmc = np.zeros((3, ndim))
    q = np.zeros((2, ndim))
    for i in range(ndim):
        r_mean = robust.mean(flat_samples[:, i])
        q_std = az.hdi(flat_samples[:, i], hdi_prob = std)
        mcmc[:, i] = np.array([q_std[0], r_mean, q_std[1]])
        q[:, i] = np.diff(mcmc[:, i])
    return mcmc, q    


def display_params(mcmc: np.ndarray, q: np.ndarray, labels: list):
    """
    Display the parameters
    
    :params:
        mmcm    : array, contains the lower limit, mean, and the upper limit of (HDI)
        q       : array, the difference between mean and lower limit, and the mean and upper limit
        labels  : list, the name of the parameters
        
    """
    
    ndim = len(mcmc.T)
    
    for i in range(ndim):
        if i != ndim - 1:
            print(f"{labels[i]} = {mcmc[:, i][1]}, -{q[:, i][0]}, +{q[:, i][1]} \n")
        else:
            print(f"{labels[i]} = {mcmc[:, i][1]}, -{q[:, i][0]}, +{q[:, i][1]}")
    

sample = run_emcee(data, params, config.nwalkers, config.iterations, log_probability, log_prior)
ndim = len(params)
kplt.plot_walkers(sample, ndim, config.burnin, labels)
flat_samples = sample.get_chain(discard = config.burnin, thin = config.thin, flat = True)
theta_params, q = get_params(flat_samples)
plot_res(data, theta_params[1], flat_samples, config.chain_size_plot, config.show_err)

if options not in multi_instruments_options and options not in combination_options:
    fig = corner.corner(flat_samples, labels = labels)
    display_params(theta_params, q, labels_)

if options == "COMBINED (SB1 + AST)":
    fig = corner.corner(flat_samples, labels = labels)
    display_params(theta_params, q, labels_)
    #a, e, k1, p, i =  flat_samples[:, 2], flat_samples[:, 3], flat_samples[:, 0], flat_samples[:, 7], flat_samples[:, 4]
    #G = 1.3271244*10**20
    #binary_mass_f = (p*k1**3)/(2*np.pi)

if options == "COMBINED (SB2 + AST)":
    corner.corner(flat_samples, labels = labels, quantiles = [0.16, 0.5, 0.84],
                  show_titles = True, title_kwargs = {"fontsize": 12},
                  max_n_ticks = 2)
    print(" ")
    print("Calculate the masses, semi-major axis and the distance")
    a, e, k1, k2, p, i =  flat_samples[:, 3], flat_samples[:, 4], flat_samples[:, 0], flat_samples[:, 1], flat_samples[:, 8], flat_samples[:, 5]
    M1sin3i = 1.036149e-7*(1 - e**2)**(3/2)*(k1 + k2)**2*k2*p
    M1 = M1sin3i/(np.sin(np.radians(i))**3)
    M2sin3i = 1.036149e-7*(1 - e**2)**(3/2)*(k1 + k2)**2*k1*p
    M2 = M2sin3i/(np.sin(np.radians(i))**3)
    a_AU = 9.191940e-5*(k1 + k2)*p*np.sqrt(1 - e**2)/(np.sin(np.radians(i)))
    d = a_AU/a
    pi = 1000*d**(-1)
    samples_m = np.column_stack((M1, M2, a_AU, d, pi))
    labels_m = [r"$M_A$", r"$M_B$", r"$a_{AU}$", "d", r"$\pi$"]
    corner.corner(samples_m, labels = labels_m, quantiles = [0.16, 0.5, 0.84],
                  show_titles = True, title_kwargs = {"fontsize": 12}, 
                  max_n_ticks = 2)
    display_params(theta_params, q, labels_)
    print(" ")
    theta_params2, q2 = get_params(samples_m)
    labels_m_ = np.array(["M_A", "M_B", "a_AU", "d", "pi"])
    display_params(theta_params2, q2, labels_m_)
    t_params = np.column_stack((theta_params, theta_params2))
    labels_save = np.hstack((labels_, labels_m_))

if options == "SB1 (multi instruments)" or options == "SB2 (multi instruments)":
    fig = corner.corner(flat_samples, labels = labels)
    display_params(theta_params, q, labels_)

if options == "COMBINED (SB1 (multi instruments) + AST)":
    fig = corner.corner(flat_samples, labels = labels)
    display_params(theta_params, q, labels_)

if options == "COMBINED (SB2 (multi instruments) + AST)":
    corner.corner(flat_samples, labels = labels, quantiles = [0.16, 0.5, 0.84],
                  show_titles = True, title_kwargs = {"fontsize": 12},
                  max_n_ticks = 2)
    print(" ")
    print("Calculate the masses, semi-major axis and the distance")
    a, e, k1, k2, p, i =  flat_samples[:, 3], flat_samples[:, 4], flat_samples[:, 0], flat_samples[:, 1], flat_samples[:, 8], flat_samples[:, 5]
    M1sin3i = 1.036149e-7*(1 - e**2)**(3/2)*(k1 + k2)**2*k2*p
    M1 = M1sin3i/(np.sin(np.radians(i))**3)
    M2sin3i = 1.036149e-7*(1 - e**2)**(3/2)*(k1 + k2)**2*k1*p
    M2 = M2sin3i/(np.sin(np.radians(i))**3)
    a_AU = 9.191940e-5*(k1 + k2)*p*np.sqrt(1 - e**2)/(np.sin(np.radians(i)))
    d = a_AU/a
    pi = 1000*d**(-1)
    samples_m = np.column_stack((M1, M2, a_AU, d, pi))
    labels_m = [r"$M_A$", r"$M_B$", r"$a_{AU}$", "d", r"$\pi$"]
    corner.corner(samples_m, labels = labels_m, quantiles = [0.16, 0.5, 0.84],
                  show_titles = True, title_kwargs = {"fontsize": 12}, 
                  max_n_ticks = 2)
    display_params(theta_params, q, labels_)
    print(" ")
    theta_params2, q2 = get_params(samples_m)
    labels_m_ = np.array(["M_A", "M_B", "a_AU", "d", "pi"])
    display_params(theta_params2, q2, labels_m_)
    t_params = np.column_stack((theta_params, theta_params2))
    labels_save = np.hstack((labels_, labels_m_))
