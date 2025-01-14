# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:57:00 2023

@author: Jonatan Rudrasingam

Keplerian Orbit for Visual and Spectroscopic Binary 2 (KOVSB2)

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
import corner
#import svboppy as svboppy
import arviz as az
import robust

plt.close("all")

######################################################################################################################
# MCMC options
######################################################################################################################
nwalkers = 32 #64
iterations = 2000 #5000
burnin = 750
#options = "SB1"
options = "SB2"
#options = "AST"
#options = "COMBINED"

######################################################################################################################
# Input files
######################################################################################################################
rv_file_name = 'test.npy' #'/home/lakeclean/Documents/speciale/rv_data/test.npy' #"rv_sep_data.npy"
rv_params_file_name = 'test_params.npy'  #'/home/lakeclean/Documents/speciale/rv_data/test_params.npy' #""rv_params_chi_dra.npy""

if options == "SB1":
    time_rv, rva, rva_err = np.load(rv_file_name, allow_pickle = True)
    data_rv = (time_rv, rva, rva_err)
    params_rv = np.load("rv_params_chi_dra_sb1.npy", allow_pickle = True)
    k1, t0, e, w, p, v01 = params_rv
    labels_rv = [r"$k_A$", r"$t_0$", r"$e$", r"$\omega\,$", r"$p$", r"$\gamma$"]
    labels_rv_ = ["kA", "t0", "e", "w", "p", "v0"]

if options == "SB2" or options == "COMBINED":
    time_rv, rva, rvb, rva_err, rvb_err = np.load(rv_file_name, allow_pickle = True)
    #rms_a, rms_b = 0.11308804776140002, 0.1359807697844086
    #rva_err, rvb_err = rms_a*np.ones(len(rva)), rms_b*np.ones(len(rvb))
    data_rv = (time_rv, time_rv, rva, rvb, rva_err, rvb_err)
    params_rv = np.load(rv_params_file_name, allow_pickle = True)
    k1, k2, t0, e, w, p, v01, v02 = params_rv
    labels_rv = [r"$k_A$", r"$k_B$", r"$t_0$", r"$e$", r"$\omega\,$", r"$p$", r"$\gamma\,_{A}$", r"$\gamma\,_{B}$"]
    labels_rv_ = ["kA", "kB", "t0", "e", "w", "p", "v0_A", "v0_B"]

if options == "AST" or options == "COMBINED":
    time_ast, rho, theta, rho_err, theta_err = np.load("ast_pa_sep5.npy", allow_pickle = True)
    # Calculate the correlation coefficient between theta and rho
    c = np.corrcoef((rho, theta))[0][1]
    data_ast = (time_ast - 2457000, rho, theta, rho_err, theta_err)
    a = 0.1244
    i = 74 #74.42 # 74.8
    W = 230.35 #230.30 #50.5
    _, t0, e, w, p, _ = np.load("rv_params_chi_dra_sb1.npy", allow_pickle = True)
    params_ast = np.hstack((t0, a, e, i, w, W, p))
    labels_ast = [r"$t_0$", r"$a$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$"]
    labels_ast_ = ["t0", "a", "e", "i", "w", "W", "p"]

if options == "COMBINED":
    data = (time_rv, time_rv, rva, rvb, rva_err, rvb_err, time_ast - 2457000, rho, theta, rho_err, 10*theta_err)
    params = np.hstack((k1, k2, t0, a, e, i, w, W, p, v01, v02))
    labels = [r"$k_A$", r"$k_B$", r"$t_0$", r"$a$", r"$e$", r"$i$", r"$\omega\,$", r"$\Omega\,$", r"$p$", r"$\gamma\,_{A}$", r"$\gamma\,_{B}$"]
    labels_ = np.array(["kA", "kB", "t0", "a", "e", "i", "w", "W", "p", "v0_A", "v0_B"])

######################################################################################################################
# Priors
######################################################################################################################
# Uniform priors
t0_lim = [-np.inf, np.inf] # [-np.inf, np.inf] 
k_a_lim = [-100, 100]  # [-np.inf, np.inf]  
k_b_lim = [-100, 100]  # [-np.inf, np.inf]  
a_lim = [0, np.inf]  # [0, np.inf]  
e_lim = [0, 1] # [0, 1] 
i_lim = [0, 180] # [0, 180]  
w_lim = [0, 360] # [0, 360] 
W_lim = [0, 360] # [0, 360]
p_lim = [1,200] # [-np.inf, np.inf]  
v0_a_lim = [-100, 100] # [-np.inf, np.inf]  
v0_b_lim = [-100, 100] # [-np.inf, np.inf] 

# Normal priors
if options == "SB1":
    mu = np.array([k1, t0, p, v01])
    sigma = np.array([k1*0.01, t0*0.015, 0.5, v01*0.01])

if options == "SB2":
    mu = np.array([k1, k2, t0, p, v01, v02])
    sigma = np.array([k1*0.01, k2*0.01, t0*0.015, 0.5, v01*0.01, v02*0.01])

if options == "AST":
    mu = np.array([t0, a, p])
    sigma = np.array([t0*0.015, a*0.01, 0.5])

if options == "COMBINED":
    mu = np.array([k1, k2, t0, a, p, v01, v02])
    sigma = np.array([k1*0.01, k2*0.01, t0*0.015, a*0.01, 0.5, v01*0.01, v02*0.01])

def gauss_dis(p: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    Computes the log probability from Gaussian distributions at given parameters
    Returns the sum of these log probabilities
    
    :params:
      p        : array, parameters
      mu       : array, the mean
      sigma    : array, the standard deviation
    
    :return 
      gp       : float, the log probability from Gaussian distributions

    """
    
    gp = np.sum(np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(p - mu)**2/sigma**2)
    
    return gp

######################################################################################################################
# Kepler's equation
######################################################################################################################
def solve_keplers_equation(mean_anomaly: float, eccentricity: float, 
                           tolerance: float = 1.e-5):
    """
    Solves Keplers equation for the eccentric anomaly using the Newton-Raphson methond
    Adopted from sboppy
    
    This method implements a solver for Kepler's Equation:
    .. math::
        M = E - sin(E),
    following Charles and Tatum, Celestial Mechanics and Dynamical Astronomy, vol.69, p.357 (1998).
    M is the "mean anomaly" and E is the "eccentric anomaly".
    Other ways to solve Keplers equation see Markley (Markley 1995, CeMDA, 63, 101).
    
    :params:
      mean_anomaly       : float, the mean anomaly
      eccentricity       : float, the eccentricity
      tolerance          : float, the convergence tolerance
    
    :return 
      eccentric anomaly  : float, the eccentric anomaly

    """
    
    if eccentricity == 0.:
        #  For circular orbit the mean and eccentric anomaly are equal
        return mean_anomaly

    new_eccentric_anomaly = np.pi  # first guess for the eccentric anomaly is pi
    for i in range(100):
        old_eccentric_anomaly = new_eccentric_anomaly + 0.

        new_eccentric_anomaly = (mean_anomaly - eccentricity*(old_eccentric_anomaly*np.cos(old_eccentric_anomaly) - np.sin(old_eccentric_anomaly)))/(1.0 - eccentricity*np.cos(old_eccentric_anomaly))

        if np.max(np.abs(new_eccentric_anomaly - old_eccentric_anomaly)/old_eccentric_anomaly) < tolerance:
            break

    return new_eccentric_anomaly

######################################################################################################################
# Radial velocity
######################################################################################################################
def radial_velocity(t: np.ndarray, k: float, e: float, w: float, p: float, 
                    t0: float, v0: float):
    """
    Calculate the radial velocity orbit from the orbital parameters.
    Adopted from sboppy
    
    :params:
      t             : array, the epochs
      k             : float, radial velocity semi-amplitude
      e             : float, the eccentricity
      w             : float, the argument of periastron in degrees
      p             : float, period in days
      t0            : float, reference epoch
      v0            : float, the system velocity
    
    :return:
      rad_vel + v0  : array, the model radial velocity for the given orbital parameters.
    
    """

    #  Calculate the mean anomaly and fold it by 2pi to reduces numerical errors
    #  when solving Kepler's equation for the eccentric anomaly
    mean_anomaly = 2.0*np.pi*np.remainder((t - t0)/p, 1.)
    eccentric_anomaly = solve_keplers_equation(mean_anomaly, e)
    cos_E = np.cos(eccentric_anomaly)
    sin_E = np.sin(eccentric_anomaly)

    #  Calculate true anomaly f
    cos_f = (cos_E - e)/(1.0 - e*cos_E)
    sin_f = (np.sqrt(1.0 - e**2)*sin_E)/(1.0 - e*cos_E)
    w_ = np.pi/180.*w
    #  use V = V0 + K(cos(w+f) + ecos(w))
    #  but expand the cos(w+f) so as not to do arccos(f)
    rad_vel = k*(np.cos(w_)*(e + cos_f) - np.sin(w_)*sin_f)

    # Add system velocity
    return rad_vel + v0


def log_likelihood_rv_sb1(params_rv: np.ndarray, time_rva: np.ndarray, 
                          rva: np.ndarray, rva_err: np.ndarray):
    """
    The log likelihood function for radial velocity
    
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the epochs for A
      rva              : array, radial velocity for most luminous component (A)
      rva_err          : array, error in the radial velocity for A
    
    :return:
      -0.5*chi2_rv     : float, the log likelihood for radial velocity
    
    """
    
    # Extract the radial velocity parameters
    k_a, t0, e, w, p, v0_a = params_rv
    rva_model = radial_velocity(time_rva, k_a, e, w, p, t0, v0_a)
    
    chi2_rv = np.sum(((rva - rva_model)**2)/rva_err**2)

    return -0.5*chi2_rv


def log_likelihood_rv_sb2(params_rv: np.ndarray, time_rva: np.ndarray, 
                          time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                          rva_err: np.ndarray, rvb_err: np.ndarray):
    """
    The log likelihood function for radial velocity
    
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the epochs for A
      time_rvb         : array, the epochs for B
      rva              : array, radial velocity for most luminous component (A)
      rvb              : array, radial velocity for least luminous component (B)
      rva_err          : array, error in the radial velocity for A
      rvb_err          : array, error in the radial velocity for B
    
    :return:
      -0.5*chi2_rv     : float, the log likelihood for radial velocity
    
    """
    
    # Extract the radial velocity parameters
    try:
        k_a, k_b, t0, e, w, p, v0_a, v0_b = params_rv
    except ValueError:
        k_a, k_b, t0, e, w, p, v0 = params_rv
        v0_a, v0_b = v0, v0
    w2 = np.remainder(w + 180., 360.)
    rva_model = radial_velocity(time_rva, k_a, e, w, p, t0, v0_a)
    rvb_model = radial_velocity(time_rvb, k_b, e, w2,  p, t0, v0_b)
        
    chi2_rva = np.sum(((rva - rva_model)**2)/rva_err**2)
    chi2_rvb = np.sum(((rvb - rvb_model)**2)/rvb_err**2)
    chi2_rv = chi2_rva + chi2_rvb

    return -0.5*chi2_rv


def log_prior_rv_sb1(params_rv: np.ndarray):
    """
    The log prior for radial velocity. Returns the probability for the given
    parameters.
        
    :params:
      params_rv        : array, radial velocity parameters
      
    :return:
      gp or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, t0, e, w, p, v0_a = params_rv
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1]
    
    uni_prior_rv = pri_k and pri_t0 and pri_e and pri_w and pri_p and pri_v0
    gauss_priors = np.array([k_a, t0, p, v0_a])
    
    if uni_prior_rv:
        gp = gauss_dis(gauss_priors, mu, sigma)
        return gp
    else:
        return -np.inf
    

def log_prior_rv_sb2(params_rv: np.ndarray):
    """
    The log prior for radial velocity. Returns the probability for the given
    parameters.
        
    :params:
      params_rv        : array, radial velocity parameters
      
    :return:
      gp or -np.inf    : float, probability for the given log prior
    
    """
    
    try:
        k_a, k_b, t0, e, w, p, v0_a, v0_b = params_rv
    except ValueError:
        k_a, k_b, t0, e, w, p, v0 = params_rv
        v0_a, v0_b = v0, v0
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1] and v0_b_lim[0] < v0_b < v0_b_lim[1]
    
    uni_prior_rv = pri_k and pri_t0 and pri_e and pri_w and pri_p and pri_v0
    gauss_priors = np.array([k_a, k_b, t0, p, v0_a, v0_b])
    
    if uni_prior_rv:
        gp = gauss_dis(gauss_priors, mu, sigma)
        return 0 #gp
    else:
        return -np.inf


def log_probability_rv_sb1(params_rv: np.ndarray, time_rva: np.ndarray, 
                           rva: np.ndarray, rva_err: np.ndarray):
    """
    The log probability function for radial velocity measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the epochs for A
      rva              : array, radial velocity for most luminous component (A)
      rva_err          : array, error in the radial velocity for A
    
    """
    
    lp = log_prior_rv_sb1(params_rv)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rv_sb1(params_rv, time_rva, rva, rva_err)


def log_probability_rv_sb2(params_rv: np.ndarray, time_rva: np.ndarray, 
                           time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                           rva_err: np.ndarray, rvb_err: np.ndarray):
    """
    The log probability function for radial velocity measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the epochs for A
      time_rvb         : array, the epochs for B
      rva              : array, radial velocity for most luminous component (A)
      rvb              : array, radial velocity for least luminous component (B)
      rva_err          : array, error in the radial velocity for A
      rvb_err          : array, error in the radial velocity for B
    
    """
    
    lp = log_prior_rv_sb2(params_rv)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rv_sb2(params_rv, time_rva, time_rvb, rva, rvb, rva_err, rvb_err)


def plot_rv_sb1(data_rv: tuple, params: np.ndarray):
    """
    Plots the radial velocity orbit from radial velocity measurements along with
    a model from input parameters
        
    :params:
      data_rv    : tuple, radial velocity data
      params     : array, the radial velocity parameters for [mu - sigma, mu, mu + sigma]
    
    """
    
    time_rva_data, rva, rva_err = data_rv
    k_1, t0, e, w, p, v0_1 = params[1]
    phase_a_data = np.remainder(time_rva_data, p)/p
    
    k_1_l, t0_l, e_l, w_l, p_l, v0_1_l = params[0]
    k_1_u, t0_u, e_u, w_u, p_u, v0_1_u = params[2]
    
    rvAfit = radial_velocity(time_rva_data, k_1, e, w, p, t0, v0_1)
    oca = rva - rvAfit
    
    t_rva = np.linspace(np.min(time_rva_data), np.max(time_rva_data), 1000)
    phase_a_long = np.remainder(t_rva, p)/p
    sort_a = np.argsort(phase_a_long)
    
    rva_model = radial_velocity(t_rva, k_1, e, w, p, t0, v0_1) 
    
    rva_l = radial_velocity(t_rva, k_1_l, e_l, w_l,  p_l, t0_l, v0_1_l)
    rva_u = radial_velocity(t_rva, k_1_u, e_u, w_u, p_u, t0_u, v0_1_u)
    
    plt.figure() 
    plt.plot(phase_a_long[sort_a], rva_model[sort_a], color = "olivedrab")
    plt.fill_between(phase_a_long[sort_a], rva_l[sort_a], rva_u[sort_a], 
                     color = "olivedrab", alpha = 0.1)
    plt.plot(phase_a_data, rva, '.', color = "firebrick")
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('radial velocity [km/s]')
    plt.show()
    
    plt.figure()
    plt.plot(phase_a_data, oca, '.', color = 'firebrick')
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [km/s]')
    plt.show()
    

def plot_rv_sb2(data_rv: tuple, params: np.ndarray):
    """
    Plots the radial velocity orbit from radial velocity measurements along with
    a model from input parameters
        
    :params:
      data_rv    : tuple, radial velocity data
      params     : array, the radial velocity parameters for [mu - sigma, mu, mu + sigma]
    
    """
    
    time_rva_data, time_rvb_data, rva, rvb, rva_err, rvb_err = data_rv
    try:
        k_1, k_2, t0, e, w, p, v0_1, v0_2 = params[1]
    except ValueError:
        k_1, k_2, t0, e, w, p, v0 = params[1]
        v0_1, v0_2 = v0
    w2 = np.remainder(w + 180., 360.)
    phase_a_data = np.remainder(time_rva_data, p)/p
    phase_b_data = np.remainder(time_rvb_data, p)/p
    
    try:
        k_1_l, k_2_l, t0_l, e_l, w_l, p_l, v0_1_l, v0_2_l = params[0]
        k_1_u, k_2_u, t0_u, e_u, w_u, p_u, v0_1_u, v0_2_u = params[2]
    except ValueError:
        k_1_l, k_2_l, t0_l, e_l, w_l, p_l, v0_l = params[0]
        k_1_u, k_2_u, t0_u, e_u, w_u, p_u, v0_u = params[2]
        v0_1_l, v0_2_l = v0_l, v0_l
        v0_1_u, v0_2_u = v0_u, v0_u
        
    w2_l = np.remainder(w + 180., 360.)
    w2_u = np.remainder(w + 180., 360.)
    
    rvAfit = radial_velocity(time_rva_data, k_1, e, w, p, t0, v0_1)
    rvBfit = radial_velocity(time_rvb_data, k_2, e, w2, p, t0, v0_2)
    oca = rva - rvAfit
    ocb = rvb - rvBfit
    
    t_rva = np.linspace(np.min(time_rva_data), np.max(time_rva_data), 1000)
    t_rvb = np.linspace(np.min(time_rvb_data), np.max(time_rvb_data), 1000)
    phase_a_long = np.remainder(t_rva, p)/p
    phase_b_long = np.remainder(t_rvb, p)/p
    sort_a = np.argsort(phase_a_long)
    sort_b = np.argsort(phase_b_long)
    
    rva_model = radial_velocity(t_rva, k_1, e, w, p, t0, v0_1)
    rvb_model = radial_velocity(t_rvb, k_2, e, w2, p, t0, v0_2)  
    
    rva_l = radial_velocity(t_rva, k_1_l, e_l, w_l,  p_l, t0_l, v0_1_l)
    rvb_l = radial_velocity(t_rvb, k_2_l, e_l, w2_l, p_l, t0_l, v0_2_l)  
    
    rva_u = radial_velocity(t_rva, k_1_u, e_u, w_u, p_u, t0_u, v0_1_u)
    rvb_u = radial_velocity(t_rvb, k_2_u, e_u, w2_u, p_u, t0_u, v0_2_u)  
    
    plt.figure() 
    plt.plot(phase_a_long[sort_a], rva_model[sort_a], color = "olivedrab")
    plt.plot(phase_b_long[sort_b], rvb_model[sort_b], color = "peru")
    plt.fill_between(phase_a_long[sort_a], rva_l[sort_a], rva_u[sort_a], 
                     color = "olivedrab", alpha = 0.1)
    plt.fill_between(phase_b_long[sort_b], rvb_l[sort_b], rvb_u[sort_b],
                     color = "peru", alpha = 0.1)
    plt.plot(phase_a_data, rva, '.', color = "firebrick")
    plt.plot(phase_b_data, rvb, '.', color = "royalblue")
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('radial velocity [km/s]')
    plt.show()
    
    plt.figure()
    plt.plot(phase_a_data, oca, '.', color = 'firebrick')
    plt.plot(phase_b_data, ocb, '.', color = 'royalblue')
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [km/s]')
    plt.show()

######################################################################################################################
# Astrometry
######################################################################################################################
def astrometru_coord(rho: np.ndarray, theta: np.ndarray):
    """
    Function that converts position angles and angular seperations into 
    positions. Defined such as that -y is northwards, while x is eastwards.
    
    :params:
      rho     : array, the angular seperations
      theta   : array, the position angles
    
    :return:
      x       : array, the x positions
      y       : array, the y positions
    
    """
    
    x = np.array(rho*np.sin(theta*np.pi/180))
    y = -1.0*np.array(rho*np.cos(theta*np.pi/180))
    
    return x, y


def astrometry_pos(t: np.ndarray, a: float, e: float, i: float, w: float, 
                    W: float, p: float, t0: float):
    """
    Calculate the visual astrometric orbit from the orbital elements.
    Adopted from Tokovinin's IDL program orbit.pro by Frank Grundahl
    Modified to fit with rest of the script
    
    :params:
      t      : array, the epochs
      a      : float, the projected semi-major axis
      e      : float, the eccentricity
      i      : float, the orbital inclination in degrees
      w      : float, the argument of periastron in degrees
      W      : float, longitude of the ascending node in degrees
      p      : float, period in days
      t0     : float, reference epoch
    
    :return:
      theta  : array, the position angles
      rho    : array, the angular seperations
      x      : array, the x positions
      y      : array, the y positions
    
    """

    # Calculate the mean anomaly and fold it by 2pi to reduces numerical errors
    # when solving Kepler's equation for the eccentric anomaly
    mean_anomaly = 2.0*np.pi*np.remainder((t - t0)/p, 1.)
    eccentric_anomaly = solve_keplers_equation(mean_anomaly, e)
    cos_E = np.cos(eccentric_anomaly)
    sin_E = np.sin(eccentric_anomaly)
    
    i_ = np.pi/180.*i
    w_ = np.pi/180.*w
    W_ = np.pi/180.*W
    
    # Thiele-Innes elements
    A = a*(np.cos(w_)*np.cos(W_) - np.sin(w_)*np.sin(W_)*np.cos(i_))
    B = a*(np.cos(w_)*np.sin(W_) + np.sin(w_)*np.cos(W_)*np.cos(i_))
    F = -a*(np.sin(w_)*np.cos(W_) + np.cos(w_)*np.sin(W_)*np.cos(i_))
    G = -a*(np.sin(w_)*np.sin(W_) - np.cos(w_)*np.cos(W_)*np.cos(i_))
    
    # Visual orbit
    x = cos_E - e
    y = sin_E*np.sqrt(1 - e**2)

    pos1 = (A*x + F*y) # Δδ
    pos2 = (B*x + G*y) # Δα* = Δ(αcosδ)

    # Theta and rho
    rho = np.sqrt(pos1**2 + pos2**2)
    theta = (180.0/np.pi)*np.arctan2(pos2, pos1)
    theta = np.mod((theta + 360.0), 360)
    
    # Astrometic postion. Defined such as that -y is northwards, while x is eastwards.
    x, y = pos2, -pos1

    return theta, rho, x, y


def log_likelihood_ast(params_ast: np.ndarray, time_ast: np.ndarray, 
                      rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                      theta_err: np.ndarray):
    """
    The log likelihood function for astrometry
    
    :params:
      params_ast       : array, astrometric parameters
      time_ast         : array, the epochs
      rho              : array, the angular seperation
      theta            : array, the position angle
      rho_err          : array, error in the angular seperation
      theta_err        : array, error in the position angle
    
    :return:
      -0.5*chi2_ast    : float, the log likelihood for astrometry
    
    """
    
    global c
    t0, a, e, i, w, W, p = params_ast
    # Calculate theta and rho from model
    model_ast = astrometry_pos(time_ast, a, e, i, w, W, p, t0)
    theta_model, rho_model, _, _ = model_ast
    # Calculate chi**2
    OC_1 = np.sum((theta - theta_model)**2/((1 - c**2)*theta_err**2))
    OC_2 = np.sum((rho - rho_model)**2/((1 - c**2)*rho_err**2))
    OC_3 = -2*np.sum((c*(theta - theta_model)*(rho - rho_model))/((1 - c**2)*theta_err*rho_err))
    chi2_ast = OC_1 + OC_2 + OC_3
    
    return -0.5*chi2_ast


def log_prior_ast(params_ast: np.ndarray):
    """
    The log prior for astrometry. Returns the probability for the given
    parameters.
        
    :params:
      params_ast        : array, astrometric parameters
      
    :return:
      gp or -np.inf     : float, probability for the given log prior
    
    """
    
    t0, a, e, i, w, W, p = params_ast
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_i = i_lim[0] <= i <= i_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    
    uni_prior_ast = pri_t0 and pri_a and pri_e and pri_i and pri_w and pri_W and pri_p
    gauss_priors = np.array([t0, a, p])
    
    if uni_prior_ast:
        gp = gauss_dis(gauss_priors, mu, sigma)
        return gp
    else:
        return -np.inf


def log_probability_ast(params_ast: np.ndarray, time_ast: np.ndarray, 
                        rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                        theta_err: np.ndarray):
    """
    The log probability function for the astrometric measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
     params_ast       : array, astrometric parameters
     time_ast         : array, the epochs
     rho              : array, the angular seperation
     theta            : array, the position angle
     rho_err          : array, error in the angular seperation
     theta_err        : array, error in the position angle
    
    """
    
    lp = log_prior_ast(params_ast)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err)

    
def plot_ast(data_ast: tuple, data_um: tuple, params: np.ndarray):
    """
    Plots the astrometric orbig, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data_ast   : tuple, astrometric data used in the fitting
      data_um    : tuple: all astrometric data
      params     : array, the astrometic parameters for [mu - sigma, mu, mu + sigma]
    
    """
    
    time_ast, rho, theta, rho_err, theta_err = data_ast
    alpha, delta = astrometru_coord(rho, theta)
    
    time_ast_um, rho_um, theta_um, rho_err_um, theta_err_um = data_um
    alpha_um, delta_um = astrometru_coord(rho_um, theta_um)
    
    t0, a, e, i, w, W, p = params[1]
    t0_l, a_l, e_l, i_l, w_l, W_l, p_l = params[0]
    t0_u, a_u, e_u, i_u, w_u, W_u, p_u = params[2]
    phase = np.remainder(time_ast, p)/p
    phase_um = np.remainder(time_ast_um, p)/p
    
    ast_dmodel = astrometry_pos(time_ast, a, e, i, w, W, p, t0)
    theta_dmodel, rho_dmodel, alpha_dmodel, delta_dmodel = ast_dmodel
    
    ast_dmodel_um = astrometry_pos(time_ast_um, a, e, i, w, W, p, t0)
    theta_dmodel_um, rho_dmodel_um, alpha_dmodel, delta_dmodel_um = ast_dmodel_um
    
    t_ast = np.linspace(np.min(time_ast), np.max(time_ast), 1000)
    phase_long = np.remainder(t_ast, p)/p
    sort = np.argsort(phase_long)
    
    ast_model = astrometry_pos(t_ast, a, e, i, w, W, p, t0)
    theta_model, rho_model, alpha_model, delta_model = ast_model
    
    plt.figure()
    plt.plot(alpha_model[sort], delta_model[sort], color = "black")
    theta_l, rho_l, alpha_l, delta_l = astrometry_pos(t_ast, a_l, e_l, i_l, 
                                                      w_l, W_l, p_l, t0_l)
    theta_u, rho_u, alpha_u, delta_u = astrometry_pos(t_ast, a_u, e_u, i_u, 
                                                      w_u, W_u, p_u, t0_u)
    plt.fill_between(alpha_model, delta_l, delta_u, color = "grey")
    plt.plot(alpha_um, delta_um, '.', color = "darkorange")
    plt.plot(alpha, delta, '.', color = "royalblue")
    plt.show()
    
    plt.figure()
    plt.plot(phase_long[sort], rho_model[sort], color = "black")
    plt.fill_between(phase_long[sort], rho_l[sort], rho_u[sort], color = "grey")
    plt.plot(phase_um, rho_um, '.', color = "darkorange")
    plt.plot(phase, rho, '.', color = "royalblue")
    plt.xlabel('phase [0,1]')
    plt.ylabel(r'$\rho\,$ ["]')
    plt.show()
    
    plt.figure()
    plt.plot(phase_um, rho_um - rho_dmodel_um, '.', color = "darkorange")
    plt.plot(phase, rho - rho_dmodel, '.', color = "royalblue")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C ["]')
    plt.show()
    
    plt.figure()        
    plt.plot(phase_long[sort], theta_model[sort], color = "black")
    plt.fill_between(phase_long[sort], theta_l[sort], theta_u[sort], color = "grey")
    plt.plot(phase_um, theta_um, '.', color = "darkorange")
    plt.plot(phase, theta, '.', color = "royalblue")
    plt.xlabel('phase [0,1]')
    plt.ylabel(r'$\theta\,$ [°]')
    plt.show()
    
    plt.figure()
    plt.plot(phase_um, theta_um - theta_dmodel_um, '.', color = "darkorange")
    plt.plot(phase, theta - theta_dmodel, '.', color = "royalblue")
    plt.plot([0,1], [0,0],'--k')
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    plt.xlabel('phase [0,1]')
    plt.ylabel('O - C [°]')
    plt.show()
    
######################################################################################################################
# Radial velocity + Astrometry
######################################################################################################################
def log_likelihood(params: np.ndarray, time_rva: np.ndarray, time_rvb: np.ndarray, 
                   rva: np.ndarray, rvb: np.ndarray, rva_err: np.ndarray, 
                   rvb_err: np.ndarray, time_ast: np.ndarray, 
                   rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                   theta_err: np.ndarray):
    """
    The log likelihood function for both radial velocity and astrometry
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rva            : array, the epochs for A
      time_rvb            : array, the epochs for B
      rva                 : array, radial velocity for most luminous component (A)
      rvb                 : array, radial velocity for least luminous component (B)
      rva_err             : array, error in the radial velocity for A
      rvb_err             : array, error in the radial velocity for B
      time_ast            : array, the epochs
      rho                 : array, the angular seperation
      theta               : array, the position angle
      rho_err             : array, error in the angular seperation
      theta_err           : array, error in the position angle
    
    :return:
      chi2_rv + chi2_ast  : float, the log likelihood
    
    """
    
    try:
        k_a, k_b, t0, a, e, i, w, W, p, v0_a, v0_b = params
        params_rv = np.hstack((k_a, k_b, t0, e, w, p, v0_a, v0_b))
    except ValueError:
        k_a, k_b, t0, a, e, i, w, W, p, v0 = params
        params_rv = np.hstack((k_a, k_b, t0, e, w, p, v0))
    params_ast = np.hstack((t0, a, e, i, w, W, p))
    
    chi2_rv = log_likelihood_rv_sb2(params_rv, time_rva, time_rvb, rva, rvb, rva_err, rvb_err)
    chi2_ast = log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err)
    
    return chi2_rv + chi2_ast


def log_prior(params: np.ndarray):
    """
    The log prior for both astrometry and radial velocity. 
    Returns the probability for the given parameters.
        
    :params:
      params_rv        : array, the astrometric and spectroscopic parameters
      
    :return:
      gp or -np.inf    : float, probability for the given log prior
    
    """
    
    try:
        k_a, k_b, t0, a, e, i, w, W, p, v0_a, v0_b = params
    except ValueError:
        k_a, k_b, t0, a, e, i, w, W, p, v0 = params
        v0_a, v0_b = v0, v0
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1] and v0_b_lim[0] < v0_b < v0_b_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_v0
    gauss_priors = np.array([k_a, k_b, t0, a, p, v0_a, v0_b])
    
    if uni_prior:
        gp = gauss_dis(gauss_priors, mu, sigma)
        return gp
    else:
        return -np.inf


def log_probability(params: np.ndarray, time_rva: np.ndarray, time_rvb: np.ndarray, 
                      rva: np.ndarray, rvb: np.ndarray, rva_err: np.ndarray, 
                      rvb_err: np.ndarray, time_ast: np.ndarray, 
                      rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                      theta_err: np.ndarray):
    """
    The log probability function for the astrometric measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
     params           : array, the astrometric and spectroscopic parameters
     time_rva         : array, the epochs for A
     time_rvb         : array, the epochs for B
     rva              : array, radial velocity for most luminous component (A)
     rvb              : array, radial velocity for least luminous component (B)
     rva_err          : array, error in the radial velocity for A
     rvb_err          : array, error in the radial velocity for B
     time_ast         : array, the epochs
     rho              : array, the angular seperation
     theta            : array, the position angle
     rho_err          : array, error in the angular seperation
     theta_err        : array, error in the position angle
    
    """
    
    lp = log_prior(params)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood(params, time_rva, time_rvb, rva, rvb, rva_err, 
                               rvb_err, time_ast, rho, theta, rho_err, theta_err)


def plot_all(data_m: tuple, data: tuple, params: np.ndarray):
    """
    Plots the astrometric orbig, the angular seperation and the position angle,
    using the input parameters
        
    :params:
      data_m   : tuple, radial velocities and astrometric data used in the fitting
      data     : tuple: radial velocities + all astrometric data
      params   : array, the parameters for [mu - sigma, mu, mu + sigma]
    
    """

    time_rva, time_rvb, rva, rvb, rva_err, rvb_err, time_ast, rho, theta, rho_err, theta_err = data_m
    data_rv = time_rva, time_rvb, rva, rvb, rva_err, rvb_err
    data_ast = time_ast, rho, theta, rho_err, theta_err
    try:
        k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v01_, v02_ = params[1]
        params_rv_ = np.hstack((k1_, k2_, t0_, e_, w_, p_, v01_, v02_))
    except ValueError:
        k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v0_ = params[1]
        params_rv_ = np.hstack((k1_, k2_, t0_, e_, w_, p_, v0_))
    params_ast_ = np.hstack((t0_, a_, e_, i_, w_, W_, p_))
    
    try:
        k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v01_, v02_ = params[0]
        params_rv_l = np.hstack((k1_, k2_, t0_, e_, w_, p_, v01_, v02_))
    except ValueError:
        k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v0_ = params[0]
        params_rv_l = np.hstack((k1_, k2_, t0_, e_, w_, p_, v0_))
    params_ast_l = np.hstack((t0_, a_, e_, i_, w_, W_, p_))
    
    try:
        k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v01_, v02_ = params[2]
        params_rv_u = np.hstack((k1_, k2_, t0_, e_, w_, p_, v01_, v02_))
    except ValueError:
        k1_, k2_, t0_, a_, e_, i_, w_, W_, p_, v0_ = params[2]
        params_rv_u = np.hstack((k1_, k2_, t0_, e_, w_, p_, v0_))
    params_ast_u = np.hstack((t0_, a_, e_, i_, w_, W_, p_))
    
    params_rv = np.vstack((params_rv_l, params_rv_, params_rv_u))
    params_ast = np.vstack((params_ast_l, params_ast_, params_ast_u))
    
    plot_rv_sb2(data_rv, params_rv)
    plot_ast(data_ast, data, params_ast)
    
    
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


def plot_walkers(sample: emcee.ensemble.EnsembleSampler, ndim: int, labels: list):
    """
    Plots the position of each walkers as function of the iterations. Used in 
    order to when the walkers converge, so the burnin limit can be set.
    
    :params:
        sample : ensemble.EnsembleSampler, the emcee sampler
        ndim   : int, the number of parameters
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
    

if options == "SB1":
    sample_rv = run_emcee(data_rv, params_rv, nwalkers, iterations, log_probability_rv_sb1, log_prior_rv_sb1)
    ndim = len(params_rv)
    #tau = sample_rv.get_autocorr_time()
    plot_walkers(sample_rv, ndim, labels_rv)
    flat_samples_rv = sample_rv.get_chain(discard = burnin, thin = 15, flat = True)
    theta_params_rv, q_rv = get_params(flat_samples_rv)
    plot_rv_sb1(data_rv, theta_params_rv)
    fig = corner.corner(flat_samples_rv, labels = labels_rv)#, truths = params_rv)
    display_params(theta_params_rv, q_rv, labels_rv_)

if options == "SB2":
    sample_rv = run_emcee(data_rv, params_rv, nwalkers, iterations, log_probability_rv_sb2, log_prior_rv_sb2)
    ndim = len(params_rv)
    #tau = sample_rv.get_autocorr_time()
    plot_walkers(sample_rv, ndim, labels_rv)
    flat_samples_rv = sample_rv.get_chain(discard = burnin, thin = 15, flat = True)
    theta_params_rv, q_rv = get_params(flat_samples_rv)
    plot_rv_sb2(data_rv, theta_params_rv)
    fig = corner.corner(flat_samples_rv, labels = labels_rv)#, truths = params_rv)
    plt.show()
    display_params(theta_params_rv, q_rv, labels_rv_)
    #np.save("flat_samples", flat_samples)
    #np.save("chi_dra_params", [labels_save, t_params[1], t_params[0], t_params[2]])
    
if options == "AST":
    sample_ast = run_emcee(data_ast, params_ast, nwalkers, iterations, log_probability_ast, log_prior_ast)
    ndim = len(params_ast)
    #tau = sample_ast.get_autocorr_time()
    plot_walkers(sample_ast, ndim, labels_ast)
    flat_samples_ast = sample_ast.get_chain(discard = burnin, thin = 15, flat = True) #244
    theta_params_ast, q_ast = get_params(flat_samples_ast)
    fig = corner.corner(flat_samples_ast, labels = labels_ast)#, truths = params_ast)
    display_params(theta_params_ast, q_ast, labels_ast_)
    #np.save("flat_samples", flat_samples)
    #np.save("chi_dra_params", [labels_save, t_params[1], t_params[0], t_params[2]])

if options == "COMBINED":
    sample = run_emcee(data, params, nwalkers, iterations, log_probability, log_prior)
    ndim = len(params)
    #tau = sample.get_autocorr_time()
    plot_walkers(sample, ndim, labels)
    flat_samples = sample.get_chain(discard = burnin, thin = 15, flat = True) #179
    theta_params, q = get_params(flat_samples)
    time_ast_um, rho_um, theta_um, rho_err_um, theta_err_um = np.load("ast_pa_sep_raw2.npy", allow_pickle = True)
    data_um = (time_ast_um - 2457000, rho_um, theta_um, rho_err_um, theta_err_um)
    plot_all(data, data_um, theta_params)
    corner.corner(flat_samples, labels = labels, quantiles = [0.16, 0.5, 0.84], #flat_samples, labels = labels / flat_samples[:, 0:9], labels = labels[0:9]
                  show_titles = True, title_kwargs = {"fontsize": 12},
                  max_n_ticks = 2)
    
    rv_data = np.transpose(np.array([rva, rvb, rva_err, rvb_err]))
    ast_data = np.transpose(np.array([rho, theta, rho_err, theta_err]))
    #fit = svboppy.fit_sb2(time_rv, rv_data, time_ast, ast_data, [k1, k2], a, e, i, w, W, p, t0, [v01, v02])
    #print(" ")
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
    #np.save("flat_samples", flat_samples)
    #np.save("chi_dra_params", [labels_save, t_params[1], t_params[0], t_params[2]])
