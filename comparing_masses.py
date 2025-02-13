import make_table_of_target_info as mt
import matplotlib.pyplot as plt
import numpy as np
from ophobningslov import *
from get_vizier_parameters import find_parameter
import pandas as pd
from nsstools import NssSource


tab = mt.get_table()
IDs = tab['ID'].data



################################# Functions: ###############################################
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


#############################################################################################


idx_123 = np.where(IDs == 'KIC12317678')[0]
idx_90 = np.where(IDs == 'KIC9025370')[0]

idxs = []

labelsAndlocations = {'KIC9025370':0, 'KIC12317678':1,
                      'KIC9693187':2,'KIC4914923':3}
#labelsAndlocations = {}
#for i, ID in enumerate(IDs):
#    labelsAndlocations[ID] = i

################################### Importing data: ###################################



#### My Parameters: ####


###importing RV-data
rv_M1s = tab['M1'].data
e_rv_M1s = tab['e_M1'].data
rv_M2s = tab['M2'].data
e_rv_M2s = tab['e_M1'].data
Teff = tab['Teff'].data
M1_litt = tab['M1_seis_litt']
e_M1_litt = tab['e_M1_seis_litt']


###Seisdata:

#KIC9025370
dnu_90 = np.mean([132.835,133.234,133.31])
e_dnu_90 = np.sqrt(0.190032**2 + 0.424**2 + 0.156**2)
numax_90 = np.mean([3042.71407,2940.04548])
e_numax_90 = np.sqrt(92.3771852**2 + 249.493167**2)
mt.add_value(numax_90,'numax','KIC9025370')
mt.add_value(e_numax_90,'e_numax','KIC9025370')
mt.add_value(dnu_90,'dnu','KIC9025370')
mt.add_value(e_dnu_90,'e_dnu','KIC9025370')

#KIC12317678
dnu_123 = np.mean( [63.149, 63.289])
e_dnu_123 =  np.sqrt(0.35**2 + 0.380**2)
numax_123 = 1230
e_numax_123 =  100
mt.add_value(numax_123,'numax','KIC12317678')
mt.add_value(e_numax_123,'e_numax','KIC12317678')
mt.add_value(dnu_123,'dnu','KIC12317678')
mt.add_value(e_dnu_123,'e_dnu','KIC12317678')

#KIC9693187

#KIC4914923

#KIC10454113


#### Gaia data with NStools ####
nss = pd.read_csv("Jonatan.csv")
source_index = 0 # 1, 2, 3
source = NssSource(nss, indice = source_index)
campbell = source.campbell()


#### importing info from Vizier: ####
viz_parameter_names = ['numax','__dnu_','Dnu','Per','Tperi','ecc']
viz_parameter_errors = ['e_numax','e__dnu_','e_Dnu','e_Per','e_Tperi','e_ecc']




viz_parameters = np.zeros(shape=(len(viz_parameter_names),
                                 len(labelsAndlocations)) )
viz_errors = np.zeros(shape=(len(viz_parameter_errors),
                                 len(labelsAndlocations)) )



for j, ID in enumerate(labelsAndlocations.keys()):
    tables_of_vizier = find_parameter(ID)
    for table in tables_of_vizier:
        for i, name in enumerate(viz_parameter_names):
            try:
                parameter = np.array(table[name])
                viz_parameters[i,j] = parameter[0]
            except:
                pass
    tables_of_error = find_parameter(ID,viz_parameter_errors)
    for table in tables_of_error:
        for i, name in enumerate(viz_parameter_errors):
            try:
                error = np.array(table[name])
                viz_errors[i,j] = error[0]
            except:
                pass



print(viz_parameters)
print(viz_errors)

#yerr = (temps_low,temps_high)

################################ Plotting: #######################################


offset = 1

# Plotting dnu:
fig, ax = plt.subplots(1,4)
for j, ID in enumerate(labelsAndlocations.keys()):
    
    #from vizier:
    vdnu = viz_parameters[2,j]
    e_vdnu = viz_errors[2,j]
    ax[j].errorbar(1,vdnu,e_vdnu,marker='s',capsize=2,color='red')
    ax[j].plot([-2,5],[vdnu,vdnu],alpha=0.3,ls='--',color='red')

    vDnu = viz_parameters[3,j]
    e_vDnu = viz_errors[3,j]
    ax[j].errorbar(2,vDnu,e_vDnu,marker='s',capsize=2,color='blue')
    ax[j].plot([-2,5],[vDnu,vDnu],alpha=0.3,ls='--',color='blue')

    #from my work:
    mdnu = mt.get_value('dnu',ID)
    e_mdnu = mt.get_value('e_dnu',ID)
    ax[j].errorbar(3,mdnu,e_mdnu,marker='s',capsize=2,color='green')
    ax[j].plot([-2,5],[mdnu,mdnu],alpha=0.3,ls='--',color='green')

    ax[j].set_xlabel(ID,rotation = 45)
    ax[j].set_xticks([])
    ax[j].set_xlim(-2,5)
                 
    

ax[0].set_ylabel('dnu [muHz]')

fig.tight_layout()
plt.show()






# Plotting masses:

if False:
    M_90, R_90, e_M_90, e_R_90 = scaling_relations(numax_90,e_numax_90,
                                   dnu_90 ,e_dnu_90,Teff[idx_90][0])
    M_123, R_123, e_M_123, e_R_123 = scaling_relations(numax_123,e_numax_123,
                                   dnu_123 ,e_dnu_123,Teff[idx_123][0])
    fig, ax = plt.subplots()
    ax.errorbar(0,rv_M1s[idx_123],e_rv_M1s[idx_123],capsize=2,label='RV',color='b'
                ,fmt='o')
    ax.errorbar(1,rv_M2s[idx_90],e_rv_M2s[idx_90],capsize=2,color='b',fmt='o')

    ax.errorbar(-0.1,M1_litt[idx_123],e_M1_litt[idx_123],capsize=2,label='Seis litt',
                color='green',fmt='o')
    ax.errorbar(0.9,M1_litt[idx_90],e_M1_litt[idx_90],capsize=2,color='green',
                fmt='o')

    ax.errorbar(0.1,M_123,e_M_123,capsize=2,label='Seis',color='orange',fmt='o')
    ax.errorbar(1.1, M_90,e_M_90,capsize=2,color='orange',fmt='o')



    ax.set_xticks(ticks = list(labelsAndlocations.values()),
                  labels=labelsAndlocations.keys(),rotation=45)
    fig.tight_layout()

    ax.set_xlim(-3,3)
    ax.legend()
    plt.show()

    





































