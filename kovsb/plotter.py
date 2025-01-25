# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:33:43 2024

@author: Jonatan Rudrasingam
"""

import numpy as np
import matplotlib.pyplot as plt
import orbit
import emcee

######################################################################################################################
# Radial velocity
######################################################################################################################
def plot_rv_sb1(data_rv: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                chain_size: int, show_err: bool):
    """
    Plots the radial velocity orbit from radial velocity measurements along with
    a model from input parameters
        
    :params:
      data_rv       : tuple, radial velocity data
      params        : array, the radial velocity parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """
    
    time_rva_data, rva, rva_err, _ = data_rv
    k, t0, e, w, p, v0 = params
    phase_a_data = np.remainder(time_rva_data, p)/p
    
    rvAfit = orbit.radial_velocity(time_rva_data, k, e, w, p, t0, v0)
    oca = rva - rvAfit
    
    t_rva = np.linspace(np.min(time_rva_data), np.max(time_rva_data), 2000)
    phase_a_long = np.remainder(t_rva, p)/p
    sort_a = np.argsort(phase_a_long)
    
    rva_model = orbit.radial_velocity(t_rva, k, e, w, p, t0, v0) 
    
    randin = np.random.randint(len(flat_samples), size = chain_size)
    
    plt.figure() 
    plt.plot(phase_a_long[sort_a], rva_model[sort_a], color = "olivedrab")
    
    if chain_size != 0:
        for ind in randin:
            k_i, t0_i, e_i, w_i, p_i, v0_i = flat_samples[ind]
            phase_a_long_i = np.remainder(t_rva, p_i)/p_i
            sort_a_i = np.argsort(phase_a_long_i)
            rva_model_i = orbit.radial_velocity(t_rva, k_i, e_i, w_i, p_i, t0_i, v0_i) 
            plt.plot(phase_a_long_i[sort_a_i], rva_model_i[sort_a_i], color = "olivedrab", alpha = 0.1)
    
    if show_err == True:
        plt.errorbar(phase_a_data, rva, yerr = rva_err, fmt = '.', color = "firebrick")
    else:
        plt.plot(phase_a_data, rva, '.', color = "firebrick")
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('radial velocity [km/s]')
    plt.show()
    
    plt.figure()
    if show_err == True:
        plt.errorbar(phase_a_data, oca, yerr = rva_err, fmt = '.', color = "firebrick")
    else:
        plt.plot(phase_a_data, oca, '.', color = "firebrick")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [km/s]')
    plt.show()
    

def plot_rv_sb1_multi(data_rv: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                      chain_size: int, show_err: bool):
    """
    Plots the radial velocity orbit from radial velocity measurements (multi instruments) 
    along with a model from input parameters
        
    :params:
      data_rv       : tuple, radial velocity data
      params        : array, the radial velocity parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """
    
    time_rva_data, rva, rva_err, instruments, _ = data_rv
    k, t0, e, w, p, *v0_instrs = params
    phase_a_data = np.remainder(time_rva_data, p)/p
    
    # Center the RVs at 0
    for n, i in enumerate(instruments):
        rva[n] = rva[n] - np.array(v0_instrs)[i]
        
    
    rvAfit = orbit.radial_velocity(time_rva_data, k, e, w, p, t0, 0)
    oca = rva - rvAfit
    
    t_rva = np.linspace(np.min(time_rva_data), np.max(time_rva_data), 2000)
    phase_a_long = np.remainder(t_rva, p)/p
    sort_a = np.argsort(phase_a_long)
    
    rva_model = orbit.radial_velocity(t_rva, k, e, w, p, t0, 0) 
    
    randin = np.random.randint(len(flat_samples), size = chain_size)
    
    plt.figure() 
    plt.plot(phase_a_long[sort_a], rva_model[sort_a], color = "olivedrab")
    
    if chain_size != 0:
        for ind in randin:
            k_i, t0_i, e_i, w_i, p_i, *v0_i = flat_samples[ind]
            phase_a_long_i = np.remainder(t_rva, p_i)/p_i
            sort_a_i = np.argsort(phase_a_long_i)
            rva_model_i = orbit.radial_velocity(t_rva, k_i, e_i, w_i, p_i, t0_i, 0) 
            plt.plot(phase_a_long_i[sort_a_i], rva_model_i[sort_a_i], color = "olivedrab", alpha = 0.1)
    
    if show_err == True:
        plt.errorbar(phase_a_data, rva, yerr = rva_err, fmt = '.', color = "firebrick")
    else:
        plt.plot(phase_a_data, rva, '.', color = "firebrick")
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('radial velocity [km/s]')
    plt.show()
    
    plt.figure()
    if show_err == True:
        plt.errorbar(phase_a_data, oca, yerr = rva_err, fmt = '.', color = "firebrick")
    else:
        plt.plot(phase_a_data, oca, '.', color = "firebrick")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [km/s]')
    plt.show()
    

def plot_rv_sb2(data_rv: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                chain_size: int, show_err: bool):
    """
    Plots the radial velocity orbit from radial velocity measurements along with
    a model from input parameters
        
    :params:
      data_rv       : tuple, radial velocity data
      params        : array, the radial velocity parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """
    
    time_rva_data, time_rvb_data, rva, rvb, rva_err, rvb_err, _ = data_rv
    k_1, k_2, t0, e, w, p, v0_1, v0_2 = params
    w2 = np.remainder(w + 180., 360.)
    phase_a_data = np.remainder(time_rva_data, p)/p
    phase_b_data = np.remainder(time_rvb_data, p)/p
    
    rvAfit = orbit.radial_velocity(time_rva_data, k_1, e, w, p, t0, v0_1)
    rvBfit = orbit.radial_velocity(time_rvb_data, k_2, e, w2, p, t0, v0_2)
    oca = rva - rvAfit
    ocb = rvb - rvBfit
    
    t_rva = np.linspace(np.min(time_rva_data), np.max(time_rva_data), 2000)
    t_rvb = np.linspace(np.min(time_rvb_data), np.max(time_rvb_data), 2000)
    phase_a_long = np.remainder(t_rva, p)/p
    phase_b_long = np.remainder(t_rvb, p)/p
    sort_a = np.argsort(phase_a_long)
    sort_b = np.argsort(phase_b_long)
    
    rva_model = orbit.radial_velocity(t_rva, k_1, e, w, p, t0, v0_1)
    rvb_model = orbit.radial_velocity(t_rvb, k_2, e, w2, p, t0, v0_2)  
    
    randin = np.random.randint(len(flat_samples), size = chain_size)
    
    plt.figure() 
    plt.plot(phase_a_long[sort_a], rva_model[sort_a], color = "olivedrab")
    plt.plot(phase_b_long[sort_b], rvb_model[sort_b], color = "peru")
    
    if chain_size != 0:
        for ind in randin:
            k1_i, k2_i, t0_i, e_i, w_i, p_i, v01_i, v02_i = flat_samples[ind]
            w2_i = np.remainder(w_i + 180., 360.)
            phase_a_long_i = np.remainder(t_rva, p_i)/p_i
            phase_b_long_i = np.remainder(t_rvb, p_i)/p_i
            sort_a_i = np.argsort(phase_a_long_i)
            sort_b_i = np.argsort(phase_b_long_i)
            rva_model_i = orbit.radial_velocity(t_rva, k1_i, e_i, w_i, p_i, t0_i, v01_i) 
            rvb_model_i = orbit.radial_velocity(t_rvb, k2_i, e_i, w2_i, p_i, t0_i, v02_i) 
            plt.plot(phase_a_long_i[sort_a_i], rva_model_i[sort_a_i], color = "olivedrab", alpha = 0.1)
            plt.plot(phase_b_long_i[sort_b_i], rvb_model_i[sort_b_i], color = "peru", alpha = 0.1)
    
    if show_err == True:
        plt.errorbar(phase_a_data, rva, yerr = rva_err, fmt = '.', color = "firebrick")
        plt.errorbar(phase_b_data, rvb, yerr = rvb_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase_a_data, rva, '.', color = "firebrick")
        plt.plot(phase_b_data, rvb, '.', color = "royalblue")
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('radial velocity [km/s]')
    plt.show()
    
    plt.figure()
    if show_err == True:
        plt.errorbar(phase_a_data, oca, yerr = rva_err, fmt = '.', color = "firebrick")
        plt.errorbar(phase_b_data, ocb, yerr = rvb_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase_a_data, oca, '.', color = "firebrick")
        plt.plot(phase_b_data, ocb, '.', color = "royalblue")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [km/s]')
    plt.show()


def plot_rv_sb2_multi(data_rv: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                      chain_size: int, show_err: bool):
    """
    Plots the radial velocity orbit from radial velocity measurements (multi instruments) 
    along with model from input parameters
        
    :params:
      data_rv       : tuple, radial velocity data
      params        : array, the radial velocity parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """
    
    time_rva_data, time_rvb_data, rva, rvb, rva_err, rvb_err, instruments_a, instruments_b, _ = data_rv
    k_1, k_2, t0, e, w, p, *v0_instrs, dv2 = params
    w2 = np.remainder(w + 180., 360.)
    phase_a_data = np.remainder(time_rva_data, p)/p
    phase_b_data = np.remainder(time_rvb_data, p)/p
    
    # Center the RVs at 0
    for n, i in enumerate(instruments_a):
        rva[n] = rva[n] - np.array(v0_instrs)[i]
    for n, i in enumerate(instruments_b):
        rvb[n] = rvb[n] - np.array(v0_instrs)[i] - dv2
    
    rvAfit = orbit.radial_velocity(time_rva_data, k_1, e, w, p, t0, 0)
    rvBfit = orbit.radial_velocity(time_rvb_data, k_2, e, w2, p, t0, 0)
    oca = rva - rvAfit
    ocb = rvb - rvBfit
    
    t_rva = np.linspace(np.min(time_rva_data), np.max(time_rva_data), 2000)
    t_rvb = np.linspace(np.min(time_rvb_data), np.max(time_rvb_data), 2000)
    phase_a_long = np.remainder(t_rva, p)/p
    phase_b_long = np.remainder(t_rvb, p)/p
    sort_a = np.argsort(phase_a_long)
    sort_b = np.argsort(phase_b_long)
    
    rva_model = orbit.radial_velocity(t_rva, k_1, e, w, p, t0, 0)
    rvb_model = orbit.radial_velocity(t_rvb, k_2, e, w2, p, t0, 0)  
    
    randin = np.random.randint(len(flat_samples), size = chain_size)
    
    plt.figure() 
    plt.plot(phase_a_long[sort_a], rva_model[sort_a], color = "olivedrab")
    plt.plot(phase_b_long[sort_b], rvb_model[sort_b], color = "peru")
    
    if chain_size != 0:
        for ind in randin:
            k1_i, k2_i, t0_i, e_i, w_i, p_i, *v01_i, v02_i = flat_samples[ind]
            w2_i = np.remainder(w_i + 180., 360.)
            phase_a_long_i = np.remainder(t_rva, p_i)/p_i
            phase_b_long_i = np.remainder(t_rvb, p_i)/p_i
            sort_a_i = np.argsort(phase_a_long_i)
            sort_b_i = np.argsort(phase_b_long_i)
            rva_model_i = orbit.radial_velocity(t_rva, k1_i, e_i, w_i, p_i, t0_i, 0) 
            rvb_model_i = orbit.radial_velocity(t_rvb, k2_i, e_i, w2_i, p_i, t0_i, 0) 
            plt.plot(phase_a_long_i[sort_a_i], rva_model_i[sort_a_i], color = "olivedrab", alpha = 0.1)
            plt.plot(phase_b_long_i[sort_b_i], rvb_model_i[sort_b_i], color = "peru", alpha = 0.1)
    
    if show_err == True:
        plt.errorbar(phase_a_data, rva, yerr = rva_err, fmt = '.', color = "firebrick")
        plt.errorbar(phase_b_data, rvb, yerr = rvb_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase_a_data, rva, '.', color = "firebrick")
        plt.plot(phase_b_data, rvb, '.', color = "royalblue")
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('radial velocity [km/s]')
    plt.show()
    
    plt.figure()
    if show_err == True:
        plt.errorbar(phase_a_data, oca, yerr = rva_err, fmt = '.', color = "firebrick")
        plt.errorbar(phase_b_data, ocb, yerr = rvb_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase_a_data, oca, '.', color = "firebrick")
        plt.plot(phase_b_data, ocb, '.', color = "royalblue")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [km/s]')
    plt.show()

######################################################################################################################
# Astrometry
######################################################################################################################  
def plot_ast(data_ast: tuple, params: np.ndarray, flat_samples: np.ndarray, 
             chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data_ast      : tuple, the astrometric data
      params        : array, the astrometic parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """
    
    time_ast, rho, theta, rho_err, theta_err, _, _ = data_ast
    alpha, delta = orbit.astrometru_coord(rho, theta)
    
    t0, a, e, i, w, W, p = params
    phase = np.remainder(time_ast, p)/p
    
    ast_dmodel = orbit.astrometry_pos(time_ast, a, e, i, w, W, p, t0)
    theta_dmodel, rho_dmodel, alpha_dmodel, delta_dmodel = ast_dmodel
    
    t_ast = np.linspace(np.min(time_ast), np.max(time_ast), 2000)
    phase_long = np.remainder(t_ast, p)/p
    sort = np.argsort(phase_long)
    
    ast_model = orbit.astrometry_pos(t_ast, a, e, i, w, W, p, t0)
    theta_model, rho_model, alpha_model, delta_model = ast_model
    
    randin = np.random.randint(len(flat_samples), size = chain_size)
    
    plt.figure()
    plt.plot(alpha_model[sort], delta_model[sort], color = "black")
    
    if chain_size != 0:
        for ind in randin:
            t0_i, a_i, e_i, i_i, w_i, W_i, p_i = flat_samples[ind]
            ast_model_i = orbit.astrometry_pos(t_ast, a_i, e_i, i_i, w_i, W_i, p_i, t0_i)
            theta_model_i, rho_model_i, alpha_model_i, delta_model_i = ast_model_i
            phase_long_i = np.remainder(t_ast, p_i)/p_i
            sort_i = np.argsort(phase_long_i)
            plt.plot(alpha_model_i[sort_i], delta_model_i[sort_i], color = "black", alpha = 0.1)
    
    plt.plot(alpha, delta, '.', color = "royalblue")
    plt.xlabel('x´ ["]')
    plt.ylabel('y´ ["]')
    plt.show()
    
    plt.figure()
    plt.plot(phase_long[sort], rho_model[sort], color = "black")
    
    if chain_size != 0:
        for ind in randin:
            t0_i, a_i, e_i, i_i, w_i, W_i, p_i = flat_samples[ind]
            ast_model_i = orbit.astrometry_pos(t_ast, a_i, e_i, i_i, w_i, W_i, p_i, t0_i)
            theta_model_i, rho_model_i, alpha_model_i, delta_model_i = ast_model_i
            phase_long_i = np.remainder(t_ast, p_i)/p_i
            sort_i = np.argsort(phase_long_i)
            plt.plot(phase_long_i[sort_i], rho_model_i[sort_i], color = "black", alpha = 0.1)
    
    if show_err == True:
        plt.errorbar(phase, rho, yerr = rho_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase, rho, '.', color = "royalblue")
    plt.xlabel('phase [0,1]')
    plt.ylabel(r'$\rho\,$ ["]')
    plt.show()
    
    plt.figure()
    if show_err == True:
        plt.errorbar(phase, rho - rho_dmodel, yerr = rho_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase, rho - rho_dmodel, '.', color = "royalblue")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C ["]')
    plt.show()
    
    plt.figure()        
    plt.plot(phase_long[sort], theta_model[sort], color = "black")
    
    if chain_size != 0:
        for ind in randin:
            t0_i, a_i, e_i, i_i, w_i, W_i, p_i = flat_samples[ind]
            ast_model_i = orbit.astrometry_pos(t_ast, a_i, e_i, i_i, w_i, W_i, p_i, t0_i)
            theta_model_i, rho_model_i, alpha_model_i, delta_model_i = ast_model_i
            phase_long_i = np.remainder(t_ast, p_i)/p_i
            sort_i = np.argsort(phase_long_i)
            plt.plot(phase_long_i[sort_i], theta_model_i[sort_i], color = "black", alpha = 0.1)
    
    if show_err == True:
        plt.errorbar(phase, theta, yerr = rho_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase, theta, '.', color = "royalblue")
    plt.xlabel('phase [0,1]')
    plt.ylabel(r'$\theta\,$ [°]')
    plt.show()
    
    plt.figure()
    if show_err == True:
        plt.errorbar(phase, theta - theta_dmodel, yerr = rho_err, fmt = '.', color = "royalblue")
    else:
        plt.plot(phase, theta - theta_dmodel, '.', color = "royalblue")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [°]')
    plt.show()

######################################################################################################################
# Gaia Thiele Innes
######################################################################################################################
def plot_gti(data_ast: tuple, params: np.ndarray, flat_samples: np.ndarray, 
             chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit (Gaia Thiele Innes) using the input parameters
        
    :params:
      data_ast      : tuple, the astrometric data
      params        : array, the astrometic parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """
    
    gaia_ti, C_ma, _ = data_ast
    #_, _, alpha, delta = astrometry_pos(t0, a, e, i, w, W, p, t0)
    
    t0, a0_au, e, i, w, W, p, pi = params
    a0 = a0_au*(pi/1000)
    phase = np.remainder(t0, p)/p
    
    ast_dmodel = orbit.astrometry_pos(t0, a0, e, i, w, W, p, t0)
    theta_dmodel, rho_dmodel, alpha_dmodel, delta_dmodel = ast_dmodel
    
    t_ast = np.linspace(t0 - p, t0 + p, 2000)
    phase_long = np.remainder(t_ast, p)/p
    sort = np.argsort(phase_long)
    
    ast_model = orbit.astrometry_pos(t_ast, a0, e, i, w, W, p, t0)
    theta_model, rho_model, alpha_model, delta_model = ast_model
    
    randin = np.random.randint(len(flat_samples), size = chain_size)
    
    plt.figure()
    plt.plot(alpha_model[sort], delta_model[sort], color = "black")
    
    if chain_size != 0:
        for ind in randin:
            t0_i, a0_au_i, e_i, i_i, w_i, W_i, p_i, pi_i = flat_samples[ind]
            a0_i = a0_au_i*(pi_i/1000)
            ast_model_i = orbit.astrometry_pos(t_ast, a0_i, e_i, i_i, w_i, W_i, p_i, t0_i)
            theta_model_i, rho_model_i, alpha_model_i, delta_model_i = ast_model_i
            phase_long_i = np.remainder(t_ast, p_i)/p_i
            sort_i = np.argsort(phase_long_i)
            plt.plot(alpha_model_i[sort_i], delta_model_i[sort_i], color = "black", alpha = 0.1)
    
    #plt.plot(alpha, delta, '.', color = "royalblue")
    plt.xlabel('x´ ["]')
    plt.ylabel('y´ ["]')
    plt.show()

######################################################################################################################
# Radial velocity (SB1) + Relative Astrometry
######################################################################################################################
def plot_comb1(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
               chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rv, rva, rva_err, time_ast, rho, theta, rho_err, theta_err, c, priors = data
    data_rv = time_rv, rva, rva_err, priors
    data_ast = time_ast, rho, theta, rho_err, theta_err, c, priors
    k1_, t0_, a_, e_, i_, w_, W_, p_, v01_ = params
    params_rv = np.array([k1_, t0_, e_, w_, p_, v01_])
    params_ast = np.array([t0_, a_, e_, i_, w_, W_, p_])
    
    k1_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, v01_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, t0_s, e_s, w_s, p_s, v01_s)).T
    flat_samples_ast = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s)).T
    
    plot_rv_sb1(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_ast(data_ast, params_ast, flat_samples_ast, chain_size, show_err)

######################################################################################################################
# Radial velocity (SB1 - multi instruments) + Relative Astrometry
######################################################################################################################
def plot_comb1_multi(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                     chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rv, rva, rva_err, instrument, time_ast, rho, theta, rho_err, theta_err, c, priors = data
    data_rv = time_rv, rva, rva_err, instrument, priors
    data_ast = time_ast, rho, theta, rho_err, theta_err, c, priors
    k1_, t0_, a_, e_, i_, w_, W_, p_, *v01_ = params
    params_rv = np.concatenate((np.array([k1_]), np.array([t0_]), np.array([e_]), 
                                np.array([w_]), np.array([p_]), np.array(v01_).flatten()))
    params_ast = np.array([t0_, a_, e_, i_, w_, W_, p_])
    
    k1_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, *v01_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, t0_s, e_s, w_s, p_s, v01_s)).T
    flat_samples_ast = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s)).T
    
    plot_rv_sb1_multi(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_ast(data_ast, params_ast, flat_samples_ast, chain_size, show_err)

######################################################################################################################
# Radial velocity (SB2) + Relative Astrometry
######################################################################################################################
def plot_comb2(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
               chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rva, time_rvb, rva, rvb, rva_err, rvb_err, time_ast, rho, theta, rho_err, theta_err, c, priors = data
    data_rv = time_rva, time_rvb, rva, rvb, rva_err, rvb_err, priors
    data_ast = time_ast, rho, theta, rho_err, theta_err, c, priors
    k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v01_, v02_ = params
    params_rv = np.array([k1_, k2_, t0_, e_, w_, p_, v01_, v02_])
    params_ast =np.array([t0_, a_, e_, i_, w_, W_, p_])
    
    k1_s, k2_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, v01_s, v02_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, k2_s, t0_s, e_s, w_s, p_s, v01_s, v02_s)).T
    flat_samples_ast = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s)).T
    
    plot_rv_sb2(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_ast(data_ast, params_ast, flat_samples_ast, chain_size, show_err)
    
######################################################################################################################
# Radial velocity (SB2 - multi instruments) + Relative Astrometry
######################################################################################################################
def plot_comb2_multi(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                     chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, time_ast, rho, theta, rho_err, theta_err, c, priors = data
    data_rv = time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, priors
    data_ast = time_ast, rho, theta, rho_err, theta_err, c, priors
    k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, *v01_, dv2_ = params
    params_rv = np.concatenate((np.array([k1_]), np.array([k2_]), np.array([t0_]), 
                                np.array([e_]), np.array([w_]), np.array([p_]), 
                                np.array(v01_).flatten(), np.array([dv2_])))
    params_ast = np.array([t0_, a_, e_, i_, w_, W_, p_])
    
    k1_s, k2_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, *v01_s, dv2_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, k2_s, t0_s, e_s, w_s, p_s, v01_s, dv2_s)).T
    flat_samples_ast = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s)).T
    
    plot_rv_sb2_multi(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_ast(data_ast, params_ast, flat_samples_ast, chain_size, show_err)

######################################################################################################################
# Radial velocity (SB1) + Gaia Thiele Innes
######################################################################################################################
def plot_comb3(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
               chain_size: int, show_err: bool):
    """
    Plots the RV curve (SB1) and the astrometric orbit (Gaia Thiele Innes) 
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rv, rva, rva_err, params_gaia_o, C, priors = data
    data_rv = time_rv, rva, rva_err, priors
    data_gti = params_gaia_o, C, priors
    k1_, t0_, a_, e_, i_, w_, W_, p_, v01_, par_ = params
    params_rv = np.array([k1_, t0_, e_, w_, p_, v01_])
    params_gti = np.array([t0_, a_, e_, i_, w_, W_, p_, par_])
    
    k1_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, v01_s, par_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, t0_s, e_s, w_s, p_s, v01_s)).T
    flat_samples_gti = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s, par_s)).T
    
    plot_rv_sb1(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_gti(data_gti, params_gti, flat_samples_gti, chain_size, show_err)

######################################################################################################################
# Radial velocity (SB1 - multi instruments) + Gaia Thiele Innes
######################################################################################################################
def plot_comb3_multi(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
                     chain_size: int, show_err: bool):
    """
    Plots the RV curve (SB1 - multi instruments) and the astrometric orbit (Gaia Thiele Innes) 
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rv, rva, rva_err, instrument, params_gaia_o, C, priors = data
    data_rv = time_rv, rva, rva_err, instrument, priors
    data_gti = params_gaia_o, C, priors
    k1_, t0_, a_, e_, i_, w_, W_, p_, *v01_, par_ = params
    params_rv = np.concatenate((np.array([k1_]), np.array([t0_]), np.array([e_]), 
                                np.array([w_]), np.array([p_]), np.array(v01_).flatten()))
    params_gti = np.array([t0_, a_, e_, i_, w_, W_, p_, par_])
    
    k1_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, *v01_s, par_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, t0_s, e_s, w_s, p_s, v01_s)).T
    flat_samples_gti = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s, par_s)).T
    
    plot_rv_sb1_multi(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_gti(data_gti, params_gti, flat_samples_gti, chain_size, show_err)

######################################################################################################################
# Radial velocity (SB2) + Gaia Thiele Innes
######################################################################################################################
def plot_comb4(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
               chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rva, time_rvb, rva, rvb, rva_err, rvb_err, params_gaia_o, C, priors = data
    data_rv = time_rva, time_rvb, rva, rvb, rva_err, rvb_err, priors
    data_gti = params_gaia_o, C, priors
    k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v01_, v02_, par_ = params
    params_rv = np.array([k1_, k2_, t0_, e_, w_, p_, v01_, v02_])
    params_gti = np.array([t0_, a_, e_, i_, w_, W_, p_, par_])
    
    k1_s, k2_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, v01_s, v02_s, par_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, k2_s, t0_s, e_s, w_s, p_s, v01_s, v02_s)).T
    flat_samples_gti = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s, par_s)).T
    
    plot_rv_sb2(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_gti(data_gti, params_gti, flat_samples_gti, chain_size, show_err)

######################################################################################################################
# Radial velocity (SB2 - multi instruments) + Gaia Thiele Innes
######################################################################################################################
def plot_comb4_multi(data: tuple, params: np.ndarray, flat_samples: np.ndarray, 
               chain_size: int, show_err: bool):
    """
    Plots the astrometric orbit, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data          : tuple, radial velocities and astrometric data used in the fitting
      params        : array, the astrometric and SB2 parameters
      flat_samples  : array, the radial velocity flat smaples from MCMC
      sample_size   : int, the number samples used
      show_err      : bool, show datapoints with errors
    
    """

    time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, params_gaia_o, C, priors = data
    data_rv = time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b, priors
    data_gti = params_gaia_o, C, priors
    k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, *v01_, dv2_, par_ = params
    params_rv = np.concatenate((np.array([k1_]), np.array([k2_]), np.array([t0_]), 
                                np.array([e_]), np.array([w_]), np.array([p_]), 
                                np.array(v01_).flatten(), np.array([dv2_])))
    params_gti = np.array([t0_, a_, e_, i_, w_, W_, p_, par_])
    
    k1_s, k2_s, t0_s, a_s, e_s, i_s, w_s, W_s, p_s, *v01_s, v02_s, par_s = flat_samples.T
    flat_samples_rv = np.vstack((k1_s, k2_s, t0_s, e_s, w_s, p_s, v01_s, v02_s)).T
    flat_samples_gti = np.vstack((t0_s, a_s, e_s, i_s, w_s, W_s, p_s, par_s)).T
    
    plot_rv_sb2_multi(data_rv, params_rv, flat_samples_rv, chain_size, show_err)
    plot_gti(data_gti, params_gti, flat_samples_gti, chain_size, show_err)

######################################################################################################################
# MCMC
######################################################################################################################
def plot_walkers(sample: emcee.ensemble.EnsembleSampler, ndim: int, burnin: int, 
                 labels: list):
    """
    Plots the position of each walkers as function of the iterations. Used in 
    order to when the walkers converge, so the burnin limit can be set.
    
    :params:
        sample : ensemble.EnsembleSampler, the emcee sampler
        ndim   : int, the number of parameters
        burnin : int, the MCMC burnin
        labels : list, labels for plotting
        
    """
    
    fig, axes = plt.subplots(ndim, figsize = (10, 10), sharex = True) #7
    samples = sample.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha = 0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.axvline(x = burnin, color = "orangered", linestyle = "solid")

    axes[-1].set_xlabel("Iterations")
    plt.show()


