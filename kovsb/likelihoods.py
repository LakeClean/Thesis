"""
@author: Jonatan Rudrasingam

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
import arviz as az
import robust
import orbit

plt.close("all")

######################################################################################################################
# Radial velocity
######################################################################################################################
def log_likelihood_rv_sb1(params_rv: np.ndarray, time_rv: np.ndarray, 
                          rv: np.ndarray, rv_err: np.ndarray):
    """
    The log likelihood function for radial velocity (SB1)
    
    :params:
      params_rv       : array, radial velocity parameters
      time_rv         : array, the RV epochs
      rv              : array, radial velocity
      rv_err          : array, error in the radial velocity
    
    :return:
      -0.5*chi2_rv     : float, the log likelihood for radial velocity
    
    """
    
    # Extract the radial velocity parameters
    k_a, t0, e, w, p, v0_a = params_rv
    rv_model = orbit.radial_velocity(time_rv, k_a, e, w, p, t0, v0_a)
    
    chi2_rv = np.sum(((rv - rv_model)**2)/rv_err**2)

    return -0.5*chi2_rv


def log_likelihood_rv_sb2(params_rv: np.ndarray, time_rva: np.ndarray, 
                          time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                          rva_err: np.ndarray, rvb_err: np.ndarray):
    """
    The log likelihood function for radial velocity (SB2)
    
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the epochs for A
      time_rvb         : array, the epochs for B
      rva              : array, radial velocity for component A
      rvb              : array, radial velocity for component B
      rva_err          : array, error in the radial velocity for A
      rvb_err          : array, error in the radial velocity for B
    
    :return:
      -0.5*chi2_rv     : float, the log likelihood for radial velocity
    
    """
    
    # Extract the radial velocity parameters
    k_a, k_b, t0, e, w, p, v0_a, v0_b = params_rv
    w2 = np.remainder(w + 180., 360.)
    rva_model = orbit.radial_velocity(time_rva, k_a, e, w, p, t0, v0_a)
    rvb_model = orbit.radial_velocity(time_rvb, k_b, e, w2,  p, t0, v0_b)
        
    chi2_rva = np.sum(((rva - rva_model)**2)/rva_err**2)
    chi2_rvb = np.sum(((rvb - rvb_model)**2)/rvb_err**2)
    chi2_rv = chi2_rva + chi2_rvb

    return -0.5*chi2_rv


def log_prior_rv_sb1(params_rv: np.ndarray, priors_rv: np.ndarray):
    """
    The log prior for radial velocity (SB1). Returns the probability for the given
    parameters.
        
    :params:
      params_rv        : array, radial velocity (SB1) parameters
      priors_rv        : array, radial velocity (SB1) priors
      
    :return:
      0.0 or -np.inf   : float, probability for the given log prior
    
    """
    
    k_a, t0, e, w, p, v0 = params_rv
    k_a_lim, t0_lim, e_lim, w_lim, p_lim, v0_lim = priors_rv
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_lim[0] < v0 < v0_lim[1]
    
    uni_prior_rv = pri_k and pri_t0 and pri_e and pri_w and pri_p and pri_v0
    
    if uni_prior_rv:
        return 0.0
    else:
        return -np.inf
    

def log_prior_rv_sb2(params_rv: np.ndarray, priors_rv: np.ndarray):
    """
    The log prior for radial velocity (SB2). Returns the probability for the given
    parameters.
        
    :params:
      params_rv        : array, radial velocity (SB2) parameters
      priors_rv        : array, radial velocity (SB2) priors
      
    :return:
      gp or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, k_b, t0, e, w, p, v0_a, v0_b = params_rv
    k_a_lim, k_b_lim, t0_lim, e_lim, w_lim, p_lim, v0_a_lim, v0_b_lim = priors_rv
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1] and v0_b_lim[0] < v0_b < v0_b_lim[1]
    
    uni_prior_rv = pri_k and pri_t0 and pri_e and pri_w and pri_p and pri_v0
    
    if uni_prior_rv:
        return 0.0
    else:
        return -np.inf


def log_probability_rv_sb1(params_rv: np.ndarray, time_rv: np.ndarray, 
                           rv: np.ndarray, rv_err: np.ndarray, 
                           priors_rv: np.ndarray):
    """
    The log probability function for radial velocity measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the RV epochs
      rv               : array, radial velocity
      rv_err           : array, error in the radial velocity
      priors_rv        : array, radial velocity (SB1) priors
    
    """
    
    lp = log_prior_rv_sb1(params_rv, priors_rv)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rv_sb1(params_rv, time_rv, rv, rv_err)


def log_probability_rv_sb2(params_rv: np.ndarray, time_rva: np.ndarray, 
                           time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                           rva_err: np.ndarray, rvb_err: np.ndarray, 
                           priors_rv: np.ndarray):
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
      priors_rv        : array, radial velocity (SB2) priors
    
    """
    
    lp = log_prior_rv_sb2(params_rv, priors_rv)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rv_sb2(params_rv, time_rva, time_rvb, rva, rvb, rva_err, rvb_err)

######################################################################################################################
# Radial velocity (multi instruments)
######################################################################################################################
def log_likelihood_rv_sb1_multi(params_rv: np.ndarray, time_rv: np.ndarray, 
                                rv: np.ndarray, rv_err: np.ndarray, instrument: np.ndarray):
    """
    The log likelihood function for radial velocity (SB1)
    
    :params:
      params_rv        : array, radial velocity parameters
      time_rv          : array, the RV epochs
      rv               : array, radial velocity for most luminous component (A)
      rv_err           : array, error in the radial velocity for A
      instrument       : array, the used instruments for the given epoch
    
    :return:
      -0.5*chi2_rv     : float, the log likelihood for radial velocity
    
    """
    
    # Extract the radial velocity parameters
    k_a, t0, e, w, p, *v0_a = params_rv
    chi2_rv = 0
    for i in np.arange(len(v0_a)):
        indx = np.where(instrument == i)[0]
        rv_model = orbit.radial_velocity(time_rv[indx], k_a, e, w, p, t0, v0_a[i])
    
        chi2_rv += np.sum(((rv[indx] - rv_model)**2)/rv_err[indx]**2)

    return -0.5*chi2_rv


def log_likelihood_rv_sb2_multi(params_rv: np.ndarray, time_rva: np.ndarray, 
                                time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                                rva_err: np.ndarray, rvb_err: np.ndarray, 
                                instrument_a: np.ndarray, instrument_b: np.ndarray):
    """
    The log likelihood function for radial velocity (SB2)
    
    :params:
      params_rv        : array, radial velocity parameters
      time_rva         : array, the epochs for A
      time_rvb         : array, the epochs for B
      rva              : array, radial velocity for most luminous component (A)
      rvb              : array, radial velocity for least luminous component (B)
      rva_err          : array, error in the radial velocity for A
      rvb_err          : array, error in the radial velocity for B
      instrument_a     : array, the used instruments for the given epoch (A)
      instrument_b     : array, the used instruments for the given epoch (B)
    
    :return:
      -0.5*chi2_rv     : float, the log likelihood for radial velocity
    
    """
    
    # Extract the radial velocity parameters
    k_a, k_b, t0, e, w, p, *v0_a, dvb = params_rv
    w2 = np.remainder(w + 180., 360.)
    
    chi2_rva = 0
    for i in np.arange(len(v0_a)):
        indx = np.where(instrument_a == i)[0]
        rva_model = orbit.radial_velocity(time_rva[indx], k_a, e, w, p, t0, v0_a[i])
        chi2_rva += np.sum(((rva[indx] - rva_model)**2)/rva_err[indx]**2)
        
    chi2_rvb = 0
    for i in np.arange(len(v0_a)):
        indx = np.where(instrument_b == i)[0]
        rvb_model = orbit.radial_velocity(time_rvb[indx], k_b, e, w2,  p, t0, v0_a[i]) + dvb
        chi2_rvb += np.sum(((rvb[indx] - rvb_model)**2)/rvb_err[indx]**2)
    
    chi2_rv = chi2_rva + chi2_rvb

    return -0.5*chi2_rv


def log_prior_rv_sb1_multi(params_rv: np.ndarray, priors_rv: np.ndarray):
    """
    The log prior for radial velocity. Returns the probability for the given
    parameters.
        
    :params:
      params_rv        : array, radial velocity (SB1) parameters
      priors_rv        : array, radial velocity (SB1) priors
      
    :return:
      0.0 or -np.inf   : float, probability for the given log prior
    
    """
    
    k_a, t0, e, w, p, *v0_a = params_rv
    k_a_lim, t0_lim, e_lim, w_lim, p_lim, v0_a_lim = priors_rv
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    
    for v in v0_a:
        if v0_a_lim[0] > v or v > v0_a_lim[1]:
            return -np.inf
    
    uni_prior_rv = pri_k and pri_t0 and pri_e and pri_w and pri_p
    
    if uni_prior_rv:
        return 0.0
    else:
        return -np.inf
    

def log_prior_rv_sb2_multi(params_rv: np.ndarray, priors_rv: np.ndarray):
    """
    The log prior for radial velocity. Returns the probability for the given
    parameters.
        
    :params:
      params_rv        : array, radial velocity (SB2) parameters
      priors_rv        : array, radial velocity (SB2) priors
      
    :return:
      gp or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, k_b, t0, e, w, p, *v0_a, dvb = params_rv
    k_a_lim, k_b_lim, t0_lim, e_lim, w_lim, p_lim, v0_a_lim, dvb_lim = priors_rv
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    for v in v0_a:
        if v0_a_lim[0] > v or v > v0_a_lim[1]:
            return -np.inf
    pri_dvb = dvb_lim[0] < dvb < dvb_lim[1]
    
    uni_prior_rv = pri_k and pri_t0 and pri_e and pri_w and pri_p and pri_dvb
    
    if uni_prior_rv:
        return 0.0
    else:
        return -np.inf


def log_probability_rv_sb1_multi(params_rv: np.ndarray, time_rv: np.ndarray, 
                                 rv: np.ndarray, rv_err: np.ndarray, 
                                 instrument: np.ndarray, priors_rv: np.ndarray):
    """
    The log probability function for radial velocity measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
      params_rv        : array, radial velocity parameters
      time_rv          : array, the epochs for A
      rv               : array, radial velocity for most luminous component (A)
      rv_err           : array, error in the radial velocity for A
      instrument       : array, the used instruments for the given epoch
      priors_rv        : array, radial velocity (SB1) priors
    
    """
    
    lp = log_prior_rv_sb1_multi(params_rv, priors_rv)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rv_sb1_multi(params_rv, time_rv, rv, rv_err, instrument)


def log_probability_rv_sb2_multi(params_rv: np.ndarray, time_rva: np.ndarray, 
                                 time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                                 rva_err: np.ndarray, rvb_err: np.ndarray, 
                                 instrument_a: np.ndarray, instrument_b: np.ndarray, 
                                 priors_rv: np.ndarray):
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
      instrument_a     : array, the used instruments for the given epoch (A)
      instrument_b     : array, the used instruments for the given epoch (B)
      priors_rv        : array, radial velocity (SB1) priors
    
    """
    
    lp = log_prior_rv_sb2_multi(params_rv, priors_rv)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_rv_sb2_multi(params_rv, time_rva, time_rvb, 
                                            rva, rvb, rva_err, rvb_err, 
                                            instrument_a, instrument_b)

######################################################################################################################
# Astrometry
######################################################################################################################
def log_likelihood_ast(params_ast: np.ndarray, time_ast: np.ndarray, 
                       rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                       theta_err: np.ndarray, c: float):
    """
    The log likelihood function for astrometry
    
    :params:
      params_ast       : array, astrometric parameters
      time_ast         : array, the epochs
      rho              : array, the angular seperation
      theta            : array, the position angle
      rho_err          : array, error in the angular seperation
      theta_err        : array, error in the position angle
      c                : float, the correlation
    
    :return:
      -0.5*chi2_ast    : float, the log likelihood for astrometry
    
    """
    
    t0, a, e, i, w, W, p = params_ast
    # Calculate theta and rho from model
    model_ast = orbit.astrometry_pos(time_ast, a, e, i, w, W, p, t0)
    theta_model, rho_model, _, _ = model_ast
    # Calculate chi**2
    OC_1 = np.sum((theta - theta_model)**2/((1 - c**2)*theta_err**2))
    OC_2 = np.sum((rho - rho_model)**2/((1 - c**2)*rho_err**2))
    OC_3 = -2*np.sum((c*(theta - theta_model)*(rho - rho_model))/((1 - c**2)*theta_err*rho_err))
    chi2_ast = OC_1 + OC_2 + OC_3
    
    return -0.5*chi2_ast


def log_prior_ast(params_ast: np.ndarray, priors_ast: np.ndarray):
    """
    The log prior for astrometry. Returns the probability for the given
    parameters.
        
    :params:
      params_ast        : array, astrometric parameters
      priors_ast        : array, the relative astrometric priors
      
    :return:
      0.0 or -np.inf     : float, probability for the given log prior
    
    """
    
    t0, a, e, i, w, W, p = params_ast
    t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim = priors_ast
    
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_i = i_lim[0] <= i <= i_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    
    uni_prior_ast = pri_t0 and pri_a and pri_e and pri_i and pri_w and pri_W and pri_p
    
    if uni_prior_ast:
        return 0.0
    else:
        return -np.inf


def log_probability_ast(params_ast: np.ndarray, time_ast: np.ndarray, 
                        rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                        theta_err: np.ndarray, c: float, priors_ast: np.ndarray):
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
     c                : float, the correlation
     priors_ast       : array, the relative astrometric priors
    
    """
    
    lp = log_prior_ast(params_ast, priors_ast)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err, c)

######################################################################################################################
# Gaia Thiele Innes
######################################################################################################################
def log_likelihood_gti(params_gaia_c: np.ndarray, params_gaia_o: np.ndarray, C: np.ndarray):
    """
    The log likelihood function for astrometry
    
    :params:
      params_gaia_c     : array, the calculated Gaia parameters
      params_gaia_o     : array, the observed Gaia parameters
      C                 : array, the correlation matrix
    
    :return:
      chi2              : float, the log likelihood for astrometry
    
    """
    
    t0, a, e, i, w, W, p, par = params_gaia_c
    
    # Calculate the Thiele Innes constants
    A_model, B_model, F_model, G_model = orbit.thiele_innes(a, e, i, w, W, p, t0)
    params_gaia_model = np.array([par, A_model, B_model, F_model, G_model, e, p, t0])
    
    theta = params_gaia_model - params_gaia_o
    #the = -1/2*(np.dot(np.dot(theta.T, np.linalg.inv(C)),  theta))**2
    the = -1/2*np.dot(theta, np.linalg.solve(C, theta))

    # Calculate log likelihood
    det_C = np.linalg.det(C)
    chi2 = np.log(1/(np.sqrt((2*np.pi)**8*np.abs(det_C)))) + the
    
    return chi2


def log_prior_gti(params_gti: np.ndarray, prios_gti: np.ndarray):
    """
    The log prior for astrometry (Gaia Thiele Innes). 
    Returns the probability for the given parameters.
        
    :params:
      params_gti        : array, the calculated Gaia parameters
      prios_gti         : array, Gaia priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """

    t0, a0, e, i, w, W, p, par = params_gti
    a = a0*par
    t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, par_lim = prios_gti
    
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_i = i_lim[0] <= i <= i_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_par = par_lim[0] < par < par_lim[1]
    
    uni_prior_ast = pri_a and pri_e and pri_i and pri_w and pri_W and pri_par
    
    if uni_prior_ast:
        return 0.0
    else:
        return -np.inf


def log_probability_gti(params_c: np.ndarray, params_gaia_o: np.ndarray, 
                        C: np.ndarray, prios_gti):
    """
    The log probability function for the astrometric measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
      params_gaia_c     : array, the calculated Gaia parameters
      params_gaia_o     : array, the observed Gaia parameters
      C                 : array, the correlation matrix
      prios_gti         : array, Gaia priors
    
    """
    
    lp = log_prior_gti(params_c, prios_gti)

    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_gti(params_c, params_gaia_o, C)

######################################################################################################################
# Radial velocity (SB1) + Relative Astrometry
######################################################################################################################
def log_likelihood_comb1(params: np.ndarray, time_rv: np.ndarray, 
                         rva: np.ndarray, rva_err: np.ndarray, 
                         time_ast: np.ndarray, 
                         rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                         theta_err: np.ndarray, c: float):
    """
    The log likelihood function for both radial velocity (SB1) and astrometry
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rv             : array, the RV epochs
      rva                 : array, radial velocity for most luminous component (A)
      rva_err             : array, error in the radial velocity for A
      time_ast            : array, the epochs
      rho                 : array, the angular seperation
      theta               : array, the position angle
      rho_err             : array, error in the angular seperation
      theta_err           : array, error in the position angle
      c                   : float, the correlation
    
    :return:
      chi2_rv + chi2_ast  : float, the log likelihood
    
    """
    
    k_a, t0, a, e, i, w, W, p, v0_a = params
    params_rv = np.hstack((k_a, t0, e, w, p, v0_a))
    params_ast = np.hstack((t0, a, e, i, w, W, p))
    
    chi2_rv = log_likelihood_rv_sb1(params_rv, time_rv, rva, rva_err)
    chi2_ast = log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err, c)
    
    return chi2_rv + chi2_ast


def log_prior_comb1(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both astrometry and radial velocity (SB1). 
    Returns the probability for the given parameters.
        
    :params:
      params_rv        : array, the astrometric and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, t0, a, e, i, w, W, p, v0_a = params
    k_a_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_v0
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb1(params: np.ndarray, time_rv: np.ndarray, rva: np.ndarray, 
                          rva_err: np.ndarray, time_ast: np.ndarray, 
                          rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                          theta_err: np.ndarray, c: float, priors: np.ndarray):
    """
    The log probability function for the astrometric measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
     params           : array, the astrometric and spectroscopic parameters
     time_rv          : array, the RV epochs
     rva              : array, radial velocity for most luminous component (A)
     rva_err          : array, error in the radial velocity for A
     time_ast         : array, the epochs
     rho              : array, the angular seperation
     theta            : array, the position angle
     rho_err          : array, error in the angular seperation
     theta_err        : array, error in the position angle
     c                : float, the correlation
     priors           : array, uniform priors
    
    """
    
    lp = log_prior_comb1(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb1(params, time_rv, rva, rva_err, 
                                     time_ast, rho, theta, rho_err, theta_err, c)

######################################################################################################################
# Radial velocity (SB1 - multi instruments) + Relative Astrometry
######################################################################################################################
def log_likelihood_comb1_multi(params: np.ndarray, time_rv: np.ndarray, 
                               rva: np.ndarray, rva_err: np.ndarray, 
                               instrument: np.ndarray, time_ast: np.ndarray, 
                               rho: np.ndarray, theta: np.ndarray, 
                               rho_err: np.ndarray, theta_err: np.ndarray, c: float):
    """
    The log likelihood function for both radial velocity and astrometry
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rv             : array, the RV epochs
      rva                 : array, radial velocity for most luminous component (A)
      rva_err             : array, error in the radial velocity for A
      instrument          : array, the used instruments for the given epoch
      time_ast            : array, the epochs
      rho                 : array, the angular seperation
      theta               : array, the position angle
      rho_err             : array, error in the angular seperation
      theta_err           : array, error in the position angle
      c                   : float, the correlation
    
    :return:
      chi2_rv + chi2_ast  : float, the log likelihood
    
    """
    
    k_a, t0, a, e, i, w, W, p, *v0_a= params
    params_rv = np.hstack((k_a, t0, e, w, p, np.array(v0_a).flatten()))
    params_ast = np.hstack((t0, a, e, i, w, W, p))
    
    chi2_rv = log_likelihood_rv_sb1_multi(params_rv, time_rv, rva, rva_err, instrument)
    chi2_ast = log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err, c)
    
    return chi2_rv + chi2_ast


def log_prior_comb1_multi(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both astrometry and radial velocity (SB1). 
    Returns the probability for the given parameters.
        
    :params:
      params_rv        : array, the astrometric and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf   : float, probability for the given log prior
    
    """
    
    k_a, t0, a, e, i, w, W, p, *v0_a = params
    k_a_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    for v in v0_a:
        if v0_a_lim[0] > v or v > v0_a_lim[1]:
            return -np.inf
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb1_multi(params: np.ndarray, time_rv: np.ndarray, 
                                rva: np.ndarray, rva_err: np.ndarray, 
                                instrument: np.ndarray, time_ast: np.ndarray, 
                                rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                                theta_err: np.ndarray, c: float, priors: np.ndarray):
    """
    The log probability function for the astrometric measurements. 
    Returns the probability (lp + log_likelihood) for the given parameters 
    using log prior and the log likelihood functions.
        
    :params:
     params           : array, the astrometric and spectroscopic parameters
     time_rva         : array, the RV epochs
     rva              : array, radial velocity for most luminous component (A)
     rva_err          : array, error in the radial velocity for A
     instrument       : array, the used instruments for the given epoch
     time_ast         : array, the epochs
     rho              : array, the angular seperation
     theta            : array, the position angle
     rho_err          : array, error in the angular seperation
     theta_err        : array, error in the position angle
     c                : float, the correlation
     priors           : array, uniform priors
    
    """
    
    lp = log_prior_comb1_multi(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb1_multi(params, time_rv, rva, rva_err, instrument, 
                                           time_ast, rho, theta, rho_err, theta_err, c)

######################################################################################################################
# Radial velocity (SB2) + Relative Astrometry
######################################################################################################################
def log_likelihood_comb2(params: np.ndarray, time_rva: np.ndarray, time_rvb: np.ndarray, 
                         rva: np.ndarray, rvb: np.ndarray, rva_err: np.ndarray, 
                         rvb_err: np.ndarray, time_ast: np.ndarray, 
                         rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                         theta_err: np.ndarray, c: float):
    """
    The log likelihood function for both radial velocity and astrometry
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rva            : array, the epochs for A
      time_rvb            : array, the epochs for B
      rva                 : array, radial velocity for component A
      rvb                 : array, radial velocity for component B
      rva_err             : array, error in the radial velocity for A
      rvb_err             : array, error in the radial velocity for B
      time_ast            : array, the epochs
      rho                 : array, the angular seperation
      theta               : array, the position angle
      rho_err             : array, error in the angular seperation
      theta_err           : array, error in the position angle
      c                   : float, the correlation
    
    :return:
      chi2_rv + chi2_ast  : float, the log likelihood
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, v0_a, v0_b = params
    params_rv = np.hstack((k_a, k_b, t0, e, w, p, v0_a, v0_b))
    params_ast = np.hstack((t0, a, e, i, w, W, p))
    
    chi2_rv = log_likelihood_rv_sb2(params_rv, time_rva, time_rvb, rva, rvb, rva_err, rvb_err)
    chi2_ast = log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err, c)
    
    return chi2_rv + chi2_ast


def log_prior_comb2(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both astrometry and radial velocity (SB2). 
    Returns the probability for the given parameters.
        
    :params:
      params_rv        : array, the astrometric and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, v0_a, v0_b = params
    k_a_lim, k_b_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim, v0_b_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1] and v0_b_lim[0] < v0_b < v0_b_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_v0
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb2(params: np.ndarray, time_rva: np.ndarray, time_rvb: np.ndarray, 
                      rva: np.ndarray, rvb: np.ndarray, rva_err: np.ndarray, 
                      rvb_err: np.ndarray, time_ast: np.ndarray, 
                      rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                      theta_err: np.ndarray, c: float, priors: np.ndarray):
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
     c                : float, the correlation
     priors           : array, uniform priors
    
    """
    
    lp = log_prior_comb2(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb2(params, time_rva, time_rvb, rva, rvb, 
                                     rva_err, rvb_err, time_ast, rho, theta, 
                                     rho_err, theta_err, c)

######################################################################################################################
# Radial velocity (SB2 - multi instruments) + Relative Astrometry
######################################################################################################################
def log_likelihood_comb2_multi(params: np.ndarray, time_rva: np.ndarray, 
                               time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                               rva_err: np.ndarray, rvb_err: np.ndarray, 
                               instrument_a: np.ndarray, instrument_b: np.ndarray, 
                               time_ast: np.ndarray, rho: np.ndarray, 
                               theta: np.ndarray, rho_err: np.ndarray, theta_err: np.ndarray, 
                               c: float):
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
      instrument_a        : array, the used instruments for the given epoch (A)
      instrument_b        : array, the used instruments for the given epoch (B)
      time_ast            : array, the epochs
      rho                 : array, the angular seperation
      theta               : array, the position angle
      rho_err             : array, error in the angular seperation
      theta_err           : array, error in the position angle
      c                   : float, the correlation
    
    :return:
      chi2_rv + chi2_ast  : float, the log likelihood
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, *v0_a, dvb = params
    params_rv = np.hstack((k_a, k_b, t0, e, w, p, np.array(v0_a).flatten(), dvb))
    params_ast = np.hstack((t0, a, e, i, w, W, p))
    
    chi2_rv = log_likelihood_rv_sb2_multi(params_rv, time_rva, time_rvb, rva, rvb, rva_err, rvb_err, instrument_a, instrument_b)
    chi2_ast = log_likelihood_ast(params_ast, time_ast, rho, theta, rho_err, theta_err, c)
    
    return chi2_rv + chi2_ast


def log_prior_comb2_multi(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both astrometry and radial velocity (SB2). 
    Returns the probability for the given parameters.
        
    :params:
      params_rv        : array, the astrometric and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, *v0_a, dvb = params
    k_a_lim, k_b_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim, dvb_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    for v in v0_a:
        if v0_a_lim[0] > v or v > v0_a_lim[1]:
            return -np.inf
    pri_dvb = dvb_lim[0] < dvb < dvb_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_dvb
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb2_multi(params: np.ndarray, time_rva: np.ndarray, time_rvb: np.ndarray, 
                                rva: np.ndarray, rvb: np.ndarray, rva_err: np.ndarray, 
                                rvb_err: np.ndarray, instrument_a: np.ndarray, 
                                instrument_b: np.ndarray, time_ast: np.ndarray, 
                                rho: np.ndarray, theta: np.ndarray, rho_err: np.ndarray, 
                                theta_err: np.ndarray, c: float, priors: np.ndarray):
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
     instrument_a     : array, the used instruments for the given epoch (A)
     instrument_b     : array, the used instruments for the given epoch (B)
     time_ast         : array, the epochs
     rho              : array, the angular seperation
     theta            : array, the position angle
     rho_err          : array, error in the angular seperation
     theta_err        : array, error in the position angle
     c                : float, the correlation
     priors           : array, uniform priors
    
    """
    
    lp = log_prior_comb2_multi(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb2_multi(params, time_rva, time_rvb, rva, rvb, 
                                           rva_err, rvb_err, instrument_a, instrument_b, 
                                           time_ast, rho, theta, rho_err, theta_err, c)

######################################################################################################################
# Radial velocity (SB1) + Gaia Thiele Innes
######################################################################################################################
def log_likelihood_comb3(params: np.ndarray, time_rv: np.ndarray, rva: np.ndarray, 
                         rva_err: np.ndarray, params_gaia_o: np.ndarray, C: np.ndarray):
    """
    The log likelihood function for both radial velocity (SB1) and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rv             : array, the RV epochs
      rva                 : array, radial velocity for most luminous component (A)
      rva_err             : array, error in the radial velocity for A
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      
    
    :return:
      chi2_rv + chi2_gti  : float, the log likelihood
    
    """
    
    k_a, t0, a, e, i, w, W, p, v0_a, par = params
    params_rv = np.hstack((k_a, t0, e, w, p, v0_a))
    params_gaia_c = np.hstack((t0, a, e, i, w, W, p, par))
    
    chi2_rv = log_likelihood_rv_sb1(params_rv, time_rv, rva, rva_err)
    chi2_gti = log_likelihood_gti(params_gaia_c, params_gaia_o, C)
    
    return chi2_rv + chi2_gti


def log_prior_comb3(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both Gaia Thiele Innes and radial velocity (SB1). 
    Returns the probability for the given parameters.
        
    :params:
      params           : array, the Gaia Thiele Innes and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, t0, a, e, i, w, W, p, v0_a, par = params
    k_a_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim, par_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1]
    pri_par = par_lim[0] < par < par_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_v0 and pri_par
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb3(params: np.ndarray, time_rv: np.ndarray, rva: np.ndarray, 
                          rva_err: np.ndarray, params_gaia_o: np.ndarray, C: np.ndarray,
                          priors: np.ndarray):
    """
    The log likelihood function for both radial velocity and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rv             : array, the epochs for A
      rva                 : array, radial velocity for most luminous component (A)
      rva_err             : array, error in the radial velocity for A
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      priors              : array, priorts (SB1 + Gaia Thiele Innes)
    
    """
    
    lp = log_prior_comb3(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb3(params, time_rv, rva, rva_err, params_gaia_o, C)

######################################################################################################################
# Radial velocity (SB1 - multi instruments) + Gaia Thiele Innes
######################################################################################################################
def log_likelihood_comb3_multi(params: np.ndarray, time_rv: np.ndarray, rva: np.ndarray, 
                               rva_err: np.ndarray, instrument: np.ndarray, 
                               params_gaia_o: np.ndarray, C: np.ndarray):
    """
    The log likelihood function for both radial velocity (SB1) and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rv             : array, the RV epochs
      rva                 : array, radial velocity for most luminous component (A)
      rva_err             : array, error in the radial velocity for A
      instrument          : array, the used instruments for the given epoch
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      
    
    :return:
      chi2_rv + chi2_gti  : float, the log likelihood
    
    """
    
    k_a, t0, a, e, i, w, W, p, *v0_a, par = params
    params_rv = np.hstack((k_a, t0, e, w, p, np.array(v0_a).flatten()))
    params_gaia_c = np.hstack((t0, a, e, i, w, W, p, par))
    
    chi2_rv = log_likelihood_rv_sb1_multi(params_rv, time_rv, rva, rva_err, instrument)
    chi2_gti = log_likelihood_gti(params_gaia_c, params_gaia_o, C)
    
    return chi2_rv + chi2_gti


def log_prior_comb3_multi(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both Gaia Thiele Innes and radial velocity (SB1). 
    Returns the probability for the given parameters.
        
    :params:
      params           : array, the Gaia Thiele Innes and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, t0, a, e, i, w, W, p, *v0_a, par = params
    k_a_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim, par_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_par = par_lim[0] < par < par_lim[1]
    for v in v0_a:
        if v0_a_lim[0] > v or v > v0_a_lim[1]:
            return -np.inf
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_par
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb3_multi(params: np.ndarray, time_rv: np.ndarray, rva: np.ndarray, 
                                rva_err: np.ndarray, instrument: np.ndarray, 
                                params_gaia_o: np.ndarray, C: np.ndarray,priors: np.ndarray):
    """
    The log likelihood function for both radial velocity and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rv             : array, the epochs for A
      rva                 : array, radial velocity for most luminous component (A)
      rva_err             : array, error in the radial velocity for A
      instrument          : array, the used instruments for the given epoch
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      priors              : array, priorts (SB1 + Gaia Thiele Innes)
    
    """
    
    lp = log_prior_comb3_multi(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb3_multi(params, time_rv, rva, rva_err, instrument, params_gaia_o, C)

######################################################################################################################
# Radial velocity (SB2) + Gaia Thiele Innes
######################################################################################################################
def log_likelihood_comb4(params: np.ndarray, time_rva: np.ndarray, 
                         time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                         rva_err: np.ndarray, rvb_err: np.ndarray, 
                         params_gaia_o: np.ndarray, C: np.ndarray):
    """
    The log likelihood function for both radial velocity (SB2) and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rva            : array, the epochs for A
      time_rvb            : array, the epochs for B
      rva                 : array, radial velocity for most luminous component (A)
      rvb                 : array, radial velocity for least luminous component (B)
      rva_err             : array, error in the radial velocity for A
      rvb_err             : array, error in the radial velocity for B
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      
    
    :return:
      chi2_rv + chi2_gti  : float, the log likelihood
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, v0_a, v0_b, par = params
    params_rv = np.hstack((k_a, k_b, t0, e, w, p, v0_a, v0_b))
    params_gaia_c = np.hstack((t0, a, e, i, w, W, p, par))
    
    chi2_rv = log_likelihood_rv_sb2(params_rv, time_rva, time_rvb, rva, rvb, rva_err, rvb_err)
    chi2_gti = log_likelihood_gti(params_gaia_c, params_gaia_o, C)
    
    return chi2_rv + chi2_gti


def log_prior_comb4(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both Gaia Thiele Innes and radial velocity (SB1). 
    Returns the probability for the given parameters.
        
    :params:
      params           : array, the Gaia Thiele Innes and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, v0_a, v0_b, par = params
    k_a_lim, k_b_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim, v0_b_lim, par_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    pri_v0 = v0_a_lim[0] < v0_a < v0_a_lim[1] and v0_b_lim[0] < v0_b < v0_b_lim[1]
    pri_par = par_lim[0] < par < par_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_v0 and pri_par
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb4(params: np.ndarray, time_rva: np.ndarray, 
                          time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                          rva_err: np.ndarray, rvb_err: np.ndarray, 
                          params_gaia_o: np.ndarray, C: np.ndarray,
                          priors: np.ndarray):
    """
    The log likelihood function for both radial velocity and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rva            : array, the epochs for A
      time_rvb            : array, the epochs for B
      rva                 : array, radial velocity for most luminous component (A)
      rvb                 : array, radial velocity for least luminous component (B)
      rva_err             : array, error in the radial velocity for A
      rvb_err             : array, error in the radial velocity for B
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      priors              : array, priorts (SB1 + Gaia Thiele Innes)
    
    """
    
    lp = log_prior_comb4(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb4(params, time_rva, time_rvb, rva, rvb, 
                                     rva_err, rvb_err, params_gaia_o, C)

######################################################################################################################
# Radial velocity (SB2 - multi instruments) + Gaia Thiele Innes
######################################################################################################################
def log_likelihood_comb4_multi(params: np.ndarray, time_rva: np.ndarray, 
                               time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                               rva_err: np.ndarray, rvb_err: np.ndarray, 
                               instrument_a: np.ndarray, instrument_b: np.ndarray,
                               params_gaia_o: np.ndarray, C: np.ndarray):
    """
    The log likelihood function for both radial velocity (SB2) and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rva            : array, the epochs for A
      time_rvb            : array, the epochs for B
      rva                 : array, radial velocity for most luminous component (A)
      rvb                 : array, radial velocity for least luminous component (B)
      rva_err             : array, error in the radial velocity for A
      rvb_err             : array, error in the radial velocity for B
      instrument_a        : array, the used instruments for the given epoch (A)
      instrument_b        : array, the used instruments for the given epoch (B)
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      
    
    :return:
      chi2_rv + chi2_gti  : float, the log likelihood
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, *v0_a, dvb, par = params
    params_rv = np.hstack((k_a, k_b, t0, e, w, p, np.array(v0_a).flatten(), dvb))
    params_gaia_c = np.hstack((t0, a, e, i, w, W, p, par))
    
    chi2_rv = log_likelihood_rv_sb2_multi(params_rv, time_rva, time_rvb, rva, 
                                          rvb, rva_err, rvb_err, 
                                          instrument_a, instrument_b)
    chi2_gti = log_likelihood_gti(params_gaia_c, params_gaia_o, C)
    
    return chi2_rv + chi2_gti


def log_prior_comb4_multi(params: np.ndarray, priors: np.ndarray):
    """
    The log prior for both Gaia Thiele Innes and radial velocity (SB1). 
    Returns the probability for the given parameters.
        
    :params:
      params           : array, the Gaia Thiele Innes and spectroscopic parameters
      priors           : array, uniform priors
      
    :return:
      0.0 or -np.inf    : float, probability for the given log prior
    
    """
    
    k_a, k_b, t0, a, e, i, w, W, p, *v0_a, dvb, par = params
    k_a_lim, k_b_lim, t0_lim, a_lim, e_lim, i_lim, w_lim, W_lim, p_lim, v0_a_lim, dvb_lim, par_lim = priors
    
    pri_k = k_a_lim[0] < k_a < k_a_lim[1] and k_b_lim[0] < k_b < k_b_lim[1]
    pri_t0 = t0_lim[0] < t0 < t0_lim[1]
    pri_a = a_lim[0] < a < a_lim[1]
    pri_e = e_lim[0] <= e < e_lim[1]
    pri_w = w_lim[0] < w < w_lim[1]
    pri_W = W_lim[0] < W < W_lim[1]
    pri_p = p_lim[0] < p < p_lim[1]
    for v in v0_a:
        if v0_a_lim[0] > v or v > v0_a_lim[1]:
            return -np.inf
    pri_dvb = dvb_lim[0] < dvb < dvb_lim[1]
    pri_par = par_lim[0] < par < par_lim[1]
    
    uni_prior = pri_k and pri_t0 and pri_a and pri_e and pri_w and pri_W and pri_p and pri_dvb and pri_par
    
    if uni_prior:
        return 0.0
    else:
        return -np.inf


def log_probability_comb4_multi(params: np.ndarray, time_rva: np.ndarray, 
                                time_rvb: np.ndarray, rva: np.ndarray, rvb: np.ndarray, 
                                rva_err: np.ndarray, rvb_err: np.ndarray, 
                                instrument_a: np.ndarray, instrument_b: np.ndarray,
                                params_gaia_o: np.ndarray, C: np.ndarray, 
                                priors: np.ndarray):
    """
    The log likelihood function for both radial velocity and Gaia Thiele Innes
    
    :params:
      params              : array, the astrometric and spectroscopic parameters
      time_rva            : array, the epochs for A
      time_rvb            : array, the epochs for B
      rva                 : array, radial velocity for most luminous component (A)
      rvb                 : array, radial velocity for least luminous component (B)
      rva_err             : array, error in the radial velocity for A
      rvb_err             : array, error in the radial velocity for B
      instrument_a        : array, the used instruments for the given epoch (A)
      instrument_b        : array, the used instruments for the given epoch (B)
      params_gaia_o       : array, the observed Gaia parameters
      C                   : array, the correlation matrix
      priors              : array, priorts (SB1 + Gaia Thiele Innes)
    
    """
    
    lp = log_prior_comb4_multi(params, priors)
    if not np.isfinite(lp):
        return lp
    return lp + log_likelihood_comb4_multi(params, time_rva, time_rvb, rva, rvb, 
                                           rva_err, rvb_err, 
                                           instrument_a, instrument_b, 
                                           params_gaia_o, C)