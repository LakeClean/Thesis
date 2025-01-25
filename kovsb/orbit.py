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
#import svboppy as svboppy
import arviz as az
import robust

plt.close("all")

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


def thiele_innes(a: float, e: float, i: float, w: float, 
                 W: float, p: float, t0: float):
    """
    Calculate Thiele Innes elements from the orbital elements.
    
    :params:
      a      : float, the projected semi-major axis
      e      : float, the eccentricity
      i      : float, the orbital inclination in degrees
      w      : float, the argument of periastron in degrees
      W      : float, longitude of the ascending node in degrees
      p      : float, period in days
      t0     : float, reference epoch
    
    :return:
      A      : float, Thiele Innes element A
      B      : float, Thiele Innes element B
      F      : float, Thiele Innes element F
      G      : float, Thiele Innes element G
    
    """
    
    i_ = np.pi/180.*i
    w_ = np.pi/180.*w
    W_ = np.pi/180.*W
    
    # Thiele-Innes elements
    A = a*(np.cos(w_)*np.cos(W_) - np.sin(w_)*np.sin(W_)*np.cos(i_))
    B = a*(np.cos(w_)*np.sin(W_) + np.sin(w_)*np.cos(W_)*np.cos(i_))
    F = -a*(np.sin(w_)*np.cos(W_) + np.cos(w_)*np.sin(W_)*np.cos(i_))
    G = -a*(np.sin(w_)*np.sin(W_) - np.cos(w_)*np.cos(W_)*np.cos(i_))

    return A, B, F, G
    

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
    
    # Thiele-Innes elements
    A, B, F, G = thiele_innes(a, e, i, w, W, p, t0)
    
    # Visual orbit
    x = cos_E - e
    y = sin_E*np.sqrt(1 - e**2)

    pos1 = (A*x + F*y) # ?d
    pos2 = (B*x + G*y) # ?a* = ?(acosd)

    # Theta and rho
    rho = np.sqrt(pos1**2 + pos2**2)
    theta = (180.0/np.pi)*np.arctan2(pos2, pos1)
    theta = np.mod((theta + 360.0), 360)
    
    # Astrometic postion. Defined such as that -y is northwards, while x is eastwards.
    x, y = pos2, -pos1

    return theta, rho, x, y