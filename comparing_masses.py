import make_table_of_target_info as mt
import matplotlib.pyplot as plt
import numpy as np
from ophobningslov import *
tab = mt.get_table()


def scaling_relations(numax, e_numax,dnu,e_dnu,Teff):
    #Chaplin et al. 2014:
    numax_sun = 3090 #muHz
    dnu_sun = 135.1 #muHz
    Teff_sun = 5780 #K
    #Values in solar masses:
    varsAndVals = {'numax':[numax,e_numax],'dnu':[dnu,e_dnu]}
    M = (numax / numax_sun)**3 * (dnu/dnu_sun)**(-4) * (Teff/Teff_sun)**(3/2)
    R = (numax / numax_sun) * (dnu/dnu_sun)**(-2) * (Teff/Teff_sun)**(1/2)
    unc_M = f'(numax / {numax_sun})**3 * (dnu/{dnu_sun})**(-4) * ({Teff}/{Teff_sun})**(3/2)'
    unc_R = f'(numax / {numax_sun}) * (dnu/{dnu_sun})**(-2) * ({Teff/Teff_sun})**(1/2)'
    e_M = ophobning(unc_M,varsAndVals,False)
    e_R = ophobning(unc_R,varsAndVals,False)
    return M, R, e_M, e_R


def dnu_from_numax(numax):
    if numax < 300: #muHz
        alpha, beta = 0.259, 0.765
    else:
        alpha, beta = 0.25, 0.779
    return alpha * (numax)**beta

fig, ax = plt.subplots()
#RV data
IDs = tab['ID'].data
rv_M1s = tab['M1'].data
e_rv_M1s = tab['e_M1'].data
rv_M2s = tab['M2'].data
e_rv_M2s = tab['e_M1'].data
Teff = tab['Teff'].data
M1_litt = tab['M1_seis_litt']
e_M1_litt = tab['e_M1_seis_litt']


idx_123 = np.where(IDs == 'KIC12317678')[0]
idx_90 = np.where(IDs == 'KIC9025370')[0]



#Seisdata
dnu_123 = np.mean( [63.149, 63.289])
e_dnu_123 =  np.sqrt(0.35**2 + 0.380**2)
numax_123 = 1230
e_numax_123 =  100

dnu_90 = np.mean([132.835,133.234,133.31])
e_dnu_90 = np.sqrt(0.190032**2 + 0.424**2 + 0.156**2)
numax_90 = np.mean([3042.71407,2940.04548])
e_numax_90 = np.sqrt(92.3771852**2 + 249.493167**2)


M_90, R_90, e_M_90, e_R_90 = scaling_relations(numax_90,e_numax_90,
                                   dnu_90 ,e_dnu_90,Teff[idx_90][0])
M_123, R_123, e_M_123, e_R_123 = scaling_relations(numax_123,e_numax_123,
                                   dnu_123 ,e_dnu_123,Teff[idx_123][0])



ax.errorbar(0,rv_M1s[idx_123],e_rv_M1s[idx_123],capsize=2,label='RV',color='b'
            ,fmt='o')
ax.errorbar(1,rv_M2s[idx_90],e_rv_M2s[idx_90],capsize=2,color='b',fmt='o')

ax.errorbar(-0.1,M1_litt[idx_123],e_M1_litt[idx_123],capsize=2,label='Seis litt',
            color='green',fmt='o')
ax.errorbar(0.9,M1_litt[idx_90],e_M1_litt[idx_90],capsize=2,color='green',
            fmt='o')

ax.errorbar(0.1,M_123,e_M_123,capsize=2,label='Seis',color='orange',fmt='o')
ax.errorbar(1.1, M_90,e_M_90,capsize=2,color='orange',fmt='o')

ax.set_xlim(-3,3)
ax.legend()
plt.show()











