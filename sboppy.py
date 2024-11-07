    # -*- coding: utf-8 -*-
import astropy as ast
import os
import matplotlib.pyplot as plt
from astropy import modeling
import numpy as np
import lmfit
from astropy.io import ascii

def solve_keplers_equation(mean_anomaly, eccentricity, tolerance=1.e-5):
    """Solves Keplers equation for the eccentric anomaly using the Newton-Raphson methond

    This method implements a solver for Kepler's Equation:
    .. math::
        M = E - sin(E),
    following Charles and Tatum, Celestial Mechanics and Dynamical Astronomy, vol.69, p.357 (1998).
    M is the "mean anomaly" and E is the "eccentric anomaly".
    Other ways to solve Keplers equation see Markley (Markley 1995, CeMDA, 63, 101).

    :param mean_anomaly:
    :param eccentricity:
    :param tolerance:
    :return eccentric anomaly:

    """
    if eccentricity == 0.:
        #  For circular orbit the mean and eccentric anomaly are equal
        return mean_anomaly

    new_eccentric_anomaly = np.pi  # first guess for the eccentric anomaly is pi
    converged = False
    for i in range(100):
        old_eccentric_anomaly = new_eccentric_anomaly + 0.
        new_eccentric_anomaly = (mean_anomaly - eccentricity * (old_eccentric_anomaly * np.cos(old_eccentric_anomaly) - np.sin(old_eccentric_anomaly))) / (1.0 - eccentricity * np.cos(old_eccentric_anomaly))

        if np.max(np.abs(new_eccentric_anomaly - old_eccentric_anomaly) / old_eccentric_anomaly) < tolerance:
            converged = True
            break

    if not converged:
        print("WARNING: Calculation of the eccentric anomaly did not convergence!")

    return new_eccentric_anomaly


def radial_velocity(t, k=1., e=0., w=0, p=1., t0=0., v0=0.):
    """Calculate the orbital radial velocity from the orbital elements."""

    #  Calculate the mean anomaly and fold it by 2pi to reduces numerical errors
    #  when solving Kepler's equation for the eccentric anomaly
    mean_anomaly = 2.0 * np.pi * np.remainder((t - t0) / p, 1.)

    eccentric_anomaly = solve_keplers_equation(mean_anomaly, e)

    cos_E = np.cos(eccentric_anomaly)
    sin_E = np.sin(eccentric_anomaly)

    #  Calculate true anomaly f
    cos_f = (cos_E - e) / (1.0 - e*cos_E)
    sin_f = (np.sqrt(1.0 - e**2) * sin_E) / (1.0 - e * cos_E)

    w = np.pi / 180. * w
    #  use V = V0 + K(cos(w+f) + ecos(w))
    #  but expand the cos(w+f) so as not to do arccos(f)
    rad_vel = k*(np.cos(w) * (e + cos_f) - np.sin(w) * sin_f)

    #  Add system velocity
    return rad_vel + v0


def fit_rv_model_SB2(parameters, hjd, rv_measurements, unc=None):
    # Unpack parameter values
    k1 = parameters['k1'].value
    k2 = parameters['k2'].value

    e = parameters['e'].value
    w = parameters['w'].value
    w2 = np.remainder(w + 180., 360.) #parameters['w2'].value

    p = parameters['p'].value
    t0 = parameters['t0'].value
    try:
        v0_1 = parameters['v0_1'].value
        v0_2 = parameters['v0_2'].value
    except:
        v0_1 = parameters['v0_1'].value
        v0_2 = parameters['v0_1'].value

    res1 = radial_velocity(hjd[0], k=k1, e=e, w=w, p=p, t0=t0, v0=v0_1) - rv_measurements[0]
    res2 = radial_velocity(hjd[1], k=k2, e=e, w=w2, p=p, t0=t0, v0=v0_2) - rv_measurements[1]
    # plt.plot(hjd, radial_velocity(hjd, k=k1, e=e, omega=w, p=p, t0=t0, v0=v0))
    # plt.plot(hjd, radial_velocity(hjd, k=k2, e=e, omega=omega2, p=p, t0=t0, v0=v0))

    # plt.plot(hjd, rv_measurements[:, 0], 'k+')
    # plt.plot(hjd, rv_measurements[:, 1], 'r+')
    # plt.show()
    '''
    if unc is None:
        res = np.append(res1, res2)
    else:
        res = np.append(res1 / rv_measurements[:, 2], res2 / rv_measurements[:, 3])
    '''
    
    if len(rv_measurements)==2: #Added by Søren
        res = np.append(res1, res2)
    elif len(rv_measurements)==4:#Added by Søren
        res = np.append(res1 / rv_measurements[2], res2 / rv_measurements[3])
    else:
        print('The rv array has the wrong shape')

    return res


def fit_radvel_SB2(hjd, rv_measurements, k=[1., 1.], e=0., w=0, p=1., t0=0., v0=0., show = False):

    parameters = lmfit.Parameters()
    # parameters.add('k', value=k[0])
    parameters.add('k1', value=k[0], min=0., vary=True)
    parameters.add('k2', value=k[1], min=0., vary=True)
    parameters.add('e',  value=e, min=0., max=1., vary=True)
    parameters.add('w',  value=w, min=0., max=360., vary=True)
    parameters.add('p',  value=p, vary=True) 
    parameters.add('t0', value=t0, vary=True)

    if isinstance(v0, list):
        # use two system velocities
        parameters.add('v0_1', value=v0[0], vary=True)
        parameters.add('v0_2', value=v0[1], vary=True)
    else:
        # use one system velocities
        parameters.add('v0_1', value=v0, vary=True)

    rv_model_fit = lmfit.minimize(fit_rv_model_SB2, parameters, args=(hjd, rv_measurements), ftol=1e-12, max_nfev=1000)

    if (show == True):
        print(lmfit.fit_report(rv_model_fit, show_correl=False))

    return rv_model_fit


def fit_rv_model_SB1(parameters, hjd, rv_measurements, unc=True):
    # Unpack parameter values
    k1 = parameters['k1'].value

    e = parameters['e'].value
    w = parameters['w'].value
    w2 = np.remainder(w + 180., 360.) #parameters['w2'].value

    p = parameters['p'].value
    t0 = parameters['t0'].value
    try:
        v0_1 = parameters['v0_1'].value
        v0_2 = parameters['v0_2'].value
    except:
        v0_1 = parameters['v0_1'].value
        v0_2 = parameters['v0_1'].value

    res1 = radial_velocity(hjd, k=k1, e=e, w=w, p=p, t0=t0, v0=v0_1) - rv_measurements
    # plt.plot(hjd, radial_velocity(hjd, k=k1, e=e, omega=w, p=p, t0=t0, v0=v0))
    # plt.plot(hjd, radial_velocity(hjd, k=k2, e=e, omega=omega2, p=p, t0=t0, v0=v0))

    # plt.plot(hjd, rv_measurements[:, 0], 'k+')
    # plt.plot(hjd, rv_measurements[:, 1], 'r+')
    # plt.show()
    return res1


def fit_radvel_SB1(hjd, rv_measurements, k=[1.], e=0., w=0, p=1., t0=0., v0=0., show = False):

    parameters = lmfit.Parameters()
    # parameters.add('k', value=k[0])
    parameters.add('k1', value=k, min=0., vary=True)
    parameters.add('e',  value=e, min=0., max=1., vary=True)
    parameters.add('w',  value=w, min=0., max=360., vary=True)
    parameters.add('p',  value=p, vary=True) 
    parameters.add('t0', value=t0, vary=True)

    if isinstance(v0, list):
        # use two system velocities
        parameters.add('v0_1', value=v0[0], vary=True)
        parameters.add('v0_2', value=v0[1], vary=True)
    else:
        # use one system velocities
        parameters.add('v0_1', value=v0, vary=True)

    rv_model_fit = lmfit.minimize(fit_rv_model_SB1, parameters, args=(hjd, rv_measurements), ftol=1e-12, max_nfev=1000)

    if (show == True):
        print(lmfit.fit_report(rv_model_fit, show_correl=False))

    return rv_model_fit





if __name__ == '__main__':

    rvguess = '/mnt/hgfs/vmware-shared/Outreach/Skoleastronomi/powerpoints/E2020/V56_RVs_dis.dat'
    rvguess = 'info.txt'

    hjd, RVA, stdA, RVB, stdB = np.loadtxt(rvguess, unpack=True)
    
    """
    print( ' ###############################################' )
    print( ' Input values (hjd, rva, erva, rvb, ervb ' )
    print( ' ' )
    print(hjd)
    print( ' ' )
    print(RVA)
    print( ' ' )
    print(stdA)
    print( ' ' )
    print(RVB)
    print( ' ' )
    print(stdB)
    print( '' )
    print( ' ###############################################' )
    """

    rv_data = np.transpose(np.array([RVA,RVB,stdA,stdB]))
    t       = hjd #- 55141.03971

   
    rv_fit  = fit_radvel(t, rv_data, k=[52.6, 59.56], e=0.0, w=90.0, p=10.72, t0=56450.0, v0=-45., show = True )#=[-45., -45.])#[-45., -45.]) #-45.)#

    hjd_new = np.linspace( np.min(t), np.max(t), 1000)  # Make 1000 new time points.

    # Explicitly get the fitting results into nicely named variables
    k1 = rv_fit.params['k1'].value
    k2 = rv_fit.params['k2'].value
    e  = rv_fit.params['e'].value
    w  = rv_fit.params['w'].value
    p  = rv_fit.params['p'].value
    t0 = rv_fit.params['t0'].value
    v0 = rv_fit.params['v0_1'].value

    ########################################################
    # Let us plot the results....
    ########################################################

    ####################################################################################################################################
    # 1. figure 
    plt.plot( t,       RVA, 'k.', markersize = 15, alpha = 0.7, label = 'Primary' )
    plt.plot( t,       RVB, 'r.', markersize = 15, alpha = 0.7, label = 'Secondary' )
    plt.plot( hjd_new, radial_velocity(hjd_new, k=k1, e=e, w=w,        p=p, t0=t0, v0=v0), label = 'Model, Primary')
    plt.plot( hjd_new, radial_velocity(hjd_new, k=k2, e=e, w=w + 180., p=p, t0=t0, v0=v0), label = 'Model, Secondary')
    plt.xlim( 56425, 56575) 
    plt.ylim( -150, 50 )
    plt.xlabel( r'MJD [d]', size = 15) 
    plt.ylabel( r'Velocity [km/s]', size = 15 )
    plt.title( rvguess, size = 15 )
    plt.legend()

    ####################################################################################################################################
    # 2. figure 
    plt.figure()
    plt.plot(t, RVA - radial_velocity( t, k=k1, e=e, w=w,        p=p, t0=t0, v0=v0), 'k.', markersize = 15, alpha = 0.7, label = 'Primary' )
    plt.plot(t, RVB - radial_velocity( t, k=k2, e=e, w=w + 180., p=p, t0=t0, v0=v0), 'r.', markersize = 15, alpha = 0.7, label = 'Secondary' )
    plt.xlim( 56425, 56575) 
    plt.ylim( -4, 6 )
    plt.xlabel( r'MJD [d]', size = 15) 
    plt.ylabel( r'(O-C) [km/s]', size = 15 )
    plt.title( rvguess, size = 15 )
    plt.legend()

    ####################################################################################################################################
    # 3. figure 
    plt.figure()
    phase     = np.remainder(t, p) / p
    phase_new = np.remainder(hjd_new, p) / p
    plt.plot(phase, RVA, 'k.', markersize = 15, alpha = 0.7, label = 'Primary' )
    plt.plot(phase, RVB, 'r.', markersize = 15, alpha = 0.7, label = 'Secondary' )
    plt.scatter(phase_new, radial_velocity(hjd_new, k=k1, e=e, w=w,        p=p, t0=t0, v0=v0),marker='.', label = 'Model, Primary')
    plt.scatter(phase_new, radial_velocity(hjd_new, k=k2, e=e, w=w + 180., p=p, t0=t0, v0=v0),marker='.', label = 'Model, Secondary')
    plt.ylim( -150, 50 )
    plt.xlim( -0.1, 1.1 )
    plt.xlabel('Phase', size = 15)
    plt.ylabel('radial velocity (km/s)', size = 15)
    plt.title( rvguess, size = 15 )
    plt.legend()

    ####################################################################################################################################
    # 4. figure 
    plt.figure()
    plt.plot(phase, RVA - radial_velocity(t, k=k1, e=e, w=w,        p=p, t0=t0, v0=v0), 'k.', markersize = 15, alpha = 0.7, label = 'Primary' )
    plt.plot(phase, RVB - radial_velocity(t, k=k2, e=e, w=w + 180., p=p, t0=t0, v0=v0), 'r.', markersize = 15, alpha = 0.7, label = 'Secondary' )
    plt.xlabel('Phase', size = 15)
    plt.ylabel('(O - C) (km/s)', size = 15)
    plt.plot([0,1],[0,0],'g')	
    plt.title( rvguess, size = 15 )
    plt.xlim( -0.1, 1.1 )
    plt.ylim( -4, 6 )
    plt.legend()

    # Done with the plotting
    ####################################################################################################################################


    m1sin3i = 1.036149e-7*(1-e**2)**(3/2)*(k1+k2)**2*k2*p
    m2sin3i = 1.036149e-7*(1-e**2)**(3/2)*(k1+k2)**2*k1*p

    print( ' ------------------------------- ' )
    print( 'M1, M2 sin3i: ',  m1sin3i,m2sin3i )
    print( ' ------------------------------- ' )
    

    ### Do some 'Monte-carlo' by shifting the residuals

    k1_c = np.array([])
    k2_c = np.array([])
    e_c  = np.array([])
    w_c  = np.array([])
    p_c  = np.array([])
    t0_c = np.array([])
    v0_c = np.array([])
 

    for n in range( len(RVA) ):

     o_c     = RVA - radial_velocity(t, k=k1, e=e, w=w, p=p, t0=t0, v0=v0)
     
     RVA     = radial_velocity(t, k=k1, e=e, w=w,     p=p, t0=t0, v0=v0) + np.roll( o_c,n )
     RVB     = radial_velocity(t, k=k2, e=e, w=w+180, p=p, t0=t0, v0=v0) + np.roll( RVB - radial_velocity(t, k=k2, e=e, w=w+180, p=p, t0=t0, v0=v0), n )
     rv_data = np.transpose(np.array([RVA,RVB,stdA,stdB]))

     rv_fit  = fit_radvel(t, rv_data, k=[52.6, 59.56], e=0.0, w=90.0, p=10.72, t0=56450.0, v0=[-45., -45.])#[-45., -45.]) #-45.)#

     k1      = rv_fit.params['k1'].value
     k2      = rv_fit.params['k2'].value

     e       = rv_fit.params['e'].value
     w       = rv_fit.params['w'].value

     p       = rv_fit.params['p'].value
     t0      = rv_fit.params['t0'].value
     v0      = rv_fit.params['v0_1'].value

     k1_c    = np.append(k1_c,rv_fit.params['k1'].value)
     k2_c    = np.append(k2_c,rv_fit.params['k2'].value)

     e_c     = np.append(e_c,rv_fit.params['e'].value)
     w_c     = np.append(w_c,rv_fit.params['w'].value)

     p_c     = np.append(p_c,rv_fit.params['p'].value)
     t0_c    = np.append(t0_c,rv_fit.params['t0'].value)
     v0_c    = np.append(v0_c,rv_fit.params['v0_1'].value)

    print( ' -----------------------------------' )
    print( ' Results for residual shift fitting.' )
    print( ' -----------------------------------' )

    
    #print( k1_c, k2_c, e_c, p_c )

    m1sin3i  = 1.036149e-7*(1-e_c**2)**(3/2)*(k1_c+k2_c)**2*k2_c*p_c
    m2sin3i  = 1.036149e-7*(1-e_c**2)**(3/2)*(k1_c+k2_c)**2*k1_c*p_c
    q        = m2sin3i/m1sin3i

    #print( ' ------------------------------- ' )
    #print( 'M1, M2 sin3i: ',  m1sin3i,m2sin3i)
    #print( ' ------------------------------- ' )
    #print( m1sin3i, m2sin3i, q)


    #        Average_prim      stddev                  dev from mean                       dev from mean                           
    print( 'Primary: ' )
    print( np.mean(m1sin3i), np.std(m1sin3i), np.mean(m1sin3i)-np.min(m1sin3i), np.max(m1sin3i)-np.mean(m1sin3i) )  # Primary

    print( 'Secondary: ' )
    print( np.mean(m2sin3i), np.std(m2sin3i), np.mean(m2sin3i)-np.min(m2sin3i), np.max(m2sin3i)-np.mean(m2sin3i) )  # Secondary

    print( 'q: ' )
    print( np.mean(q),       np.std(q),       np.mean(q)-np.min(q),             np.max(q)-np.mean(q)) 
	
    print( 'k1')
    print( np.mean(k1_c), np.std(k1_c), np.mean(k1_c)-np.min(k1_c), np.max(k1_c)-np.mean(k1_c) )

    print( 'k2')
    print( np.mean(k2_c), np.std(k2_c), np.mean(k2_c)-np.min(k2_c), np.max(k2_c)-np.mean(k2_c) )


    #print( rv_fit.params['k1'] )
    #print( rv_fit.params['k1'].value )

    #print( np.degrees( [ np.mean(np.arcsin((m2sin3i/0.974)**(1.0/3.0))), np.std(np.arcsin((m2sin3i/0.974)**(1.0/3.0))), np.mean(np.arcsin((m2sin3i/0.974)**(1.0/3.0)))-np.min(np.arcsin((m2sin3i/0.974)**(1.0/3.0)))]))
