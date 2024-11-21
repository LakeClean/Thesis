import numpy as np
from astroquery.vizier import Vizier
from ophobningslov import *
import sympy as sp
#import numpy as np
'''
catalog_list = Vizier.find_catalogs('Gaia DR3')

for i in catalog_list:
    print(i)
'''
'''
GDR3 = ['2107491287660520832','2126516412237386880','2131620306552653312','2101240083023021952',
        '2135483028339642368', '6572992562249889280']


result = Vizier(row_limit=3).query_constraints(catalog=['I/357/tboasb1c',
                                                        'I/357/tbooc'],
                                               Source='2107491287660520832')
                                               #Source='2126516412237386880')

A = float(result[0]['ATI'])
B = float(result[0]['BTI'])
F = float(result[0]['FTI'])
G = float(result[0]['GTI'])

print(A,B,F,G)
'''

'I/357/tbooc' 'I/357/tboasb1c'

'''
Script for converting Thiele Innes parameters: A,B,F,G
to Campbell parameters: a,i,Omega,w

Based completely on (Halbwachs, 2022) and (Binnendijk, 1960)
uncertainty is, as described in the above, derived from differentiation
and error propagation of the given equations
'''
def find_Campbell(A,B,F,G,e_A,e_B,e_F,e_G):
    
    ############longitude of periastron and ascending node w and Omega: #######


    #value:
    Omega_w1 = np.arctan((B-F)/(A+G))  #w + Omega
    Omega_w2 = np.arctan((B+F)/(G-A))  #w - Omega
    Omega_w1_unc = '(atan((B-F)/(A+G)))'  #w + Omega
    Omega_w2_unc = '(atan((B+F)/(G-A)))'  #w - Omega
    

    if np.sign(Omega_w1) != np.sign(B-F):
        Omega_w1 += np.pi
        Omega_w1_unc += f'+{np.pi}'

    if np.sign(Omega_w2) != np.sign(-B-F):
        Omega_w2 += np.pi
        Omega_w2_unc += f'+{np.pi}'

    
    w = (Omega_w1 + Omega_w2)/2
    Omega = (Omega_w1 - Omega_w2)/2
    w_unc = f'(({Omega_w1_unc}) + ({Omega_w2_unc}))/2'
    Omega_unc = f'(({Omega_w1_unc}) - ({Omega_w2_unc}))/2'

    if Omega < 0:
        Omega += np.pi
        w += np.pi
        Omega_unc += f'+{np.pi}'
        w_unc += f'+{np.pi}'

    if Omega > np.pi:
        Omega -= np.pi
        w -= np.pi
        Omega_unc += f'-{np.pi}'
        w_unc += f'-{np.pi}'

    #uncertainty:
    varsAndVals = {'A':[A,e_A],'B':[B,e_B],'F':[F,e_F],'G':[G,e_G]}
    e_w = ophobning(w_unc,varsAndVals,False)
    e_Omega = ophobning(Omega_unc,varsAndVals,False)
        
    ############# semi major axis a: ###################

    #value:
    k = (A**2 + B**2 + F**2 + G**2)/2
    m = A*G-B*F
    j = np.sqrt(k**2 - m**2)
    a = np.sqrt(j+k)

    k_unc = '((A**2 + B**2 + F**2 + G**2)/2)'
    m_unc = '(A*G-B*F)'
    j_unc = f'(sqrt({k_unc}**2 - {m_unc}**2))'
    a_unc = f'sqrt({j_unc}+{k_unc})'
    

    #uncertainty:
    e_a = ophobning(a_unc,varsAndVals,False)
    
    ###################### Inclination i: ##########################
    #value:
    i = np.arccos(m/(a**2))
    i_unc = f'acos(({m_unc})/(({a_unc})**2))'

    #uncertainty:
    e_i = ophobning(i_unc,varsAndVals,False)

    
    return a,Omega,w,i ,e_a,e_Omega,e_w,e_i


#print(find_Campbell(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8))








