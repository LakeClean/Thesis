import make_table_of_target_info as mt
import matplotlib.pyplot as plt
import numpy as np
from ophobningslov import *
from get_vizier_parameters import find_parameter
import pandas as pd
from nsstools import NssSource

master_path = '/usr/users/au662080/'


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
                      'KIC9693187':2,'KIC4914923':3,
                      'KIC10454113':4}


################################### Writing data to dict: #############################

#KIC10454113:
'''
param_str = ['numax','dnu','numax','dnu','numax','dnu']
param = [2261,103.8, 2357.2,105.063,2337.954,105.109]
e_param = [62,1.3,8.2,0.031,32.932,0.136]
params_source = ['J/ApJS/210/1/table1','J/ApJS/210/1/table1',
                 'J/ApJ/835/173/table3','J/ApJ/835/173/table3'
                 'J/ApJS/233/23/table3','J/ApJS/233/23/table3']
'''

#Tables

tables = ['I/357/tboasb1c', 'I/357/tbooc', 'J/ApJ/835/173/table3','J/ApJS/210/1/table1',
          'J/ApJS/233/23/table3', 'J/A+A/674/A106/table1', 'J/ApJS/236/42/giants',
          'J/ApJ/844/102/table1']

table_colors = {'I/357/tboasb1c':'#377eb8',
                'I/357/tbooc':'#ff7f00',
                'J/ApJS/233/23/table3':'#dede00',
                'J/ApJ/835/173/table3':'#4daf4a',
               'J/ApJS/210/1/table1':'#f781bf',
                'J/A+A/674/A106/table1':'#a65628',
               'J/ApJS/236/42/giants': '#984ea3',
               'J/ApJ/844/102/table1':'#999999',
                'This work': '#e41a1c'}

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

CB_color_20 = ['#1F77B4','#AEC7E8','#FF7F0E','#FFBB78','#2CA02C',
             '#98DF8A','#D62728','#FF9896','#9467BD','#C5B0D5',
             '#8C564B','#C49C94','#E377C2','#F7B6D2','#7F7F7F',
             '#C7C7C7','#BCBD22','#DBDB8D','#17BECF','#9EDAE5']
CB_marker_20 = ['o','v','^','<','>','s','p','*','D','d',
                'P','X','1','2','3','4','+','x','8','$H$']


numax_types = ['numax']
e_numax_types = ['e_numax']
dnu_types = ['<dnu>', 'Dnu','Deltanu', 'Delnu','dnu']
e_dnu_types = ['e_<dnu>', 'e_Dnu','e_Deltanu', 'e_Delnu','e_dnu']

'''
table_params = [['ATI', 'e_ATI', 'BTI', 'e_BTI','FTI', 'e_FTI', 'GTI', 'e_GTI',
                  'Per', 'e_Per', 'ecc', 'e_ecc', 'Plx','e_Plx'],
                ['ATI', 'e_ATI', 'BTI', 'e_BTI','FTI', 'e_FTI', 'GTI', 'e_GTI',
                  'Per', 'e_Per', 'ecc', 'e_ecc', 'Plx','e_Plx'],
                ['numax', 'E_numax','e_numax','<dnu>','e_<dnu>','E_<dnu>'
                        ,'Teff','e_Teff','[Fe/H]','e_[Fe/H]'],
                ['numax','e_numax', 'Dnu','e_Dnu','[Fe/H]','e_[Fe/H]'],
                ['numax','e_numax','Dnu','e_Dnu'],
                ['numax','e_numax','Deltanu','e_Deltanu','logg', 'e_logg',
                        'Teff','e_Teff'],
                ['numax','e_numax','Delnu','e_Delnu','log(g)', 'e_log(g)',
                        'Teff','e_Teff','[Fe/H]','e_[Fe/H]'],
                ['numax','e_numax','dnu','e_dnu', 'plx', 'e_plx'],
                ['Plx','e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'RV', 'e_RV',
               'Teff', 'b_Teff'],
                ['RV', 'e_RV', 'VHelio', 'BC','Teff', 'logg', 'Fe/H']]

I/357/tboasb1c = ['ATI', 'e_ATI', 'BTI', 'e_BTI','FTI', 'e_FTI', 'GTI', 'e_GTI',
                  'Per', 'e_Per', 'ecc', 'e_ecc', 'Plx','e_Plx']
#['KIC9693187', 'KIC4914923']


I/357/tbooc = ['ATI', 'e_ATI', 'BTI', 'e_BTI','FTI', 'e_FTI', 'GTI', 'e_GTI',
                  'Per', 'e_Per', 'ecc', 'e_ecc', 'Plx','e_Plx']
#['KIC9025370', 'KIC12317678']


J/ApJ/835/173/table3 = ['numax', 'E_numax','e_numax','<dnu>','e_<dnu>','E_<dnu>'
                        ,'Teff','e_Teff','[Fe/H]','e_[Fe/H]']
#['KIC10454113','KIC9025370']

J/ApJS/210/1/table1 = ['numax','e_numax', 'Dnu','e_Dnu','[Fe/H]','e_[Fe/H]']
#['KIC10454113','KIC9025370']


J/ApJS/233/23/table3 = ['numax','e_numax','Dnu','e_Dnu']
#'KIC10454113','KIC9025370'


J/A+A/674/A106/table1 = ['numax','e_numax','Deltanu','e_Deltanu','logg', 'e_logg',
                        'Teff','e_Teff']

J/ApJS/236/42/giants = ['numax','e_numax','Delnu','e_Delnu','log(g)', 'e_log(g)',
                        'Teff','e_Teff','[Fe/H]','e_[Fe/H]']

J/ApJ/844/102/table1 = ['numax','e_numax','dnu','e_dnu', 'plx', 'e_plx']




I/345/gaia2 = ['Plx','e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'RV', 'e_RV',
               'Teff', 'b_Teff']


III/286/allvis = ['RV', 'e_RV', 'VHelio', 'BC','Teff', 'logg', 'Fe/H']
'''
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


###Seisdata is saved to mt table:

for i, ID in enumerate(labelsAndlocations.keys()):
    data_path = master_path + f'Speciale/data/Seismology/analysis/{ID}/'
    
    numax, e_numax = pd.read_csv(data_path + 'numax.txt').to_numpy()[0]
    dnu, e_dnu = pd.read_csv(data_path + 'alt_dnu.txt').to_numpy()[0]
    
    mt.add_value(numax,'numax',ID)
    mt.add_value(e_numax,'e_numax',ID)
    mt.add_value(dnu,'dnu',ID)
    mt.add_value(e_dnu,'e_dnu',ID)           
            

################################ Plotting: #######################################


# Plotting numax:
fig, ax = plt.subplots(1,len(labelsAndlocations.keys()),figsize=(12,5))
param_str='numax'
for j, ID in enumerate(labelsAndlocations.keys()):
    
    #from vizier:
    param_names = numax_types
    error_names = e_numax_types


    viz_parameters = []
    viz_errors = []
    tables_of_viz = find_parameter(ID,param_names+error_names)
    table_names = [] 
    
    for table_name, table in zip(tables_of_viz.keys(),tables_of_viz):
        found_param = False
        found_error = False
        
        for name in param_names:
            try:
                viz_parameters.append(float(table[name]))
                table_names.append(table_name)
                found_param = True
            except:
                pass
            
        for error in error_names:
            try:
                viz_errors.append(float(table[error]))
                found_error = True
            except:
                pass
        if (found_param == True):
            if (found_error==False):
                print('Found val but no error')

        if (found_error == True):
            if (found_param==False):
                print('Found error but no val')
            
            
    nr_params = len(viz_parameters)
    print('nr of params from litt: ', nr_params)
    off = 0
    for i in range(nr_params):
        litt_param = viz_parameters[i]
        e_litt_param = viz_errors[i]
        ax[j].errorbar(off,litt_param,e_litt_param,marker='s',capsize=2,
                       color=table_colors[table_names[i]], label=f'{table_names[i]}')
        ax[j].plot([-2,5],[litt_param,litt_param],alpha=0.3,ls='--',color='red')
        off += 0.4

    #from my work:
    param = mt.get_value(param_str,ID)
    e_param = mt.get_value('e_'+param_str,ID)
    ax[j].errorbar(off,param,e_param,marker='s',capsize=2,color='green',label='This work')
    ax[j].plot([-2,5],[param,param],alpha=0.3,ls='--',color=table_colors['This work'])

    ax[j].set_xlabel(ID)
    ax[j].set_xticks([])
    ax[j].set_xlim(-2,5)
    
ax[0].legend(loc=9, bbox_to_anchor=(0.5,-0.04),ncols=1)

ax[0].set_ylabel(param_str)
fig.subplots_adjust(bottom=0.4)

fig.tight_layout()
plt.show()


# Plotting dnu:
fig, ax = plt.subplots(1,len(labelsAndlocations.keys()),figsize=(12,5))

param_str='dnu'
for j, ID in enumerate(labelsAndlocations.keys()):
    
    #from vizier:
    param_names = dnu_types
    error_names = e_dnu_types


    viz_parameters = []
    viz_errors = []
    tables_of_viz = find_parameter(ID,param_names+error_names)
    for table in tables_of_viz:
        found_param = False
        found_error = False
        for name in param_names:
            try:
                viz_parameters.append(float(table[name]))
                found_param = True
            except:
                pass
            
        for error in error_names:
            
            try:
                viz_errors.append(float(table[error]))
                found_error = True
            except:
                pass
        if (found_param == True):
            if (found_error==False):
                print('Found val but no error')

        if (found_error == True):
            if (found_param==False):
                print('Found error but no val')
            
            
    nr_params = len(viz_parameters)
    print(nr_params)
    off = 0
    for i in range(nr_params):
        litt_param = viz_parameters[i]
        e_litt_param = viz_errors[i]
        ax[j].errorbar(off,litt_param,e_litt_param,marker='s',capsize=2,color='red')
        ax[j].plot([-2,5],[litt_param,litt_param],alpha=0.3,ls='--',color='red')
        off += 0.4




    #from my work:
    param = mt.get_value(param_str,ID)
    e_param = mt.get_value('e_'+param_str,ID)
    ax[j].errorbar(off,param,e_param,marker='s',capsize=2,color='green',label='This work')
    ax[j].plot([-2,5],[param,param],alpha=0.3,ls='--',color='green')

    ax[j].set_xlabel(ID,rotation = 45)
    ax[j].set_xticks([])
    ax[j].set_xlim(-2,5)
ax[0].legend(loc=9, bbox_to_anchor=(0.5,-0.02))

ax[0].set_ylabel(param_str)


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

    





































