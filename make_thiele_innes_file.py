import numpy as np
from astroquery.vizier import Vizier
import pandas as pd

import glob

path = '~/Speciale/data/NOT/Target_names_and_info.txt'

f = open(path).read().split('\n')[36:50]

has_T_innes = []
target_IDs = []
G_IDs = []
for line in f:
    ID = line.split('\t')[0]
    GID,val =  line.split('DR3')[-1].split('\t')

    target_IDs.append(ID.strip(' '))
    G_IDs.append(GID.strip(' '))
    has_T_innes.append(val.strip(' '))


#f.close()
path = '~/Speciale/data/thiele_innes_elements.txt'
f = open(path,'w')
f.write('Target ID, Gaia DR3 ID, A, B ,F, G, e_ATI, e_BTI, e_FTI, e_GTI\n')
for i,j,k in zip(has_T_innes,target_IDs,G_IDs):
    if i == 'yes':
        result = Vizier(row_limit=1,
                        columns = [ 'ATI', 'BTI' ,'FTI', 'GTI','e_ATI', 'e_BTI', 'e_FTI',
                            'e_GTI']).query_constraints(catalog=['I/357/tboasb1c','I/357/tbooc'],
                            Source=k)
        #print(float(result[0]['ATI']))
        A = float(result[0]['ATI'])
        B = float(result[0]['BTI'])
        F = float(result[0]['FTI'])
        G = float(result[0]['GTI'])
        A_e = float(result[0]['e_ATI'])
        B_e = float(result[0]['e_BTI'])
        F_e = float(result[0]['e_FTI'])
        G_e = float(result[0]['e_GTI'])
        
        f.write(f'{j}, {k}, {A}, {B}, {F}, {G}, {A_e}, {B_e}, {F_e}, {G_e} \n')

f.close()
