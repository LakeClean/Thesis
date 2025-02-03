import pandas as pd
from astropy.table import *
import numpy as np
from astropy.table import Table

master_path = '/usr/users/au662080'
table_path = f'{master_path}/Speciale/data/table_of_info.dat'





#data = Table(data=[target_IDs,G_IDs],names=('ID','Gaia_ID'))


def add_value(value,key,ID):
    if type(value) ==list:
        print(f'the value {value} of type {type(value)} is a bad format')
        value='NaN'
    
    dat = Table.read(table_path,format='ascii')
    ID_col = dat['ID'].data
    index = np.where(ID_col == ID)[0]

    new_col = []
    for i,line in enumerate(dat[key]):
        if index == i:
            new_col.append(value)
        else:
            new_col.append(line)

    dat.replace_column(name=key,col = new_col)
    dat.write(table_path,format='ascii',overwrite=True)

def get_value(key,ID):
    dat = Table.read(table_path,format='ascii')
    ID_col = dat['ID'].data
    value_list = dat[key].data
    index = np.where(ID_col == ID)[0]
    return value_list[index][0]


def new_column(key,col_type='object',index=-1,desc=''):
    values=np.full(14,'NaN',dtype=col_type)
    dat = Table.read(table_path,format='ascii')
    dat.add_column(values,name=key,index=index)
    dat.write(table_path,format='ascii',overwrite=True)
    
    desc_path = f'{master_path}/Speciale/data/table_of_info_description.txt'
    lines = open(desc_path).read()
    f = open(desc_path,'w')
    f.write(lines)
    f.write(f'{key}: {desc}\n')
    f.close()


    

def get_table():
    dat = Table.read(table_path,format='ascii')
    return dat



def rm_column(key):
    dat = Table.read(table_path,format='ascii')
    dat.remove_column(name=key)
    dat.write(table_path,format='ascii',overwrite=True)

'''
ID_path = '/home/lakeclean/Documents/speciale/NOT/Target_names_and_info.txt'


f = open(ID_path).read().split('\n')[36:50]

has_T_innes = []
target_IDs = []
G_IDs = []
for line in f:
    ID = line.split('\t')[0]
    GID,val =  line.split('DR3')[-1].split('\t')

    target_IDs.append(ID.strip(' '))
    G_IDs.append(GID.strip(' '))
    has_T_innes.append(val.strip(' '))
    
tab = get_table()
colnames = tab.colnames
new_tab = Table()

new_tab.add_column(np.array(target_IDs,dtype='object'),name='ID')
new_tab.add_column(np.array(G_IDs,dtype='object'),name='Gaia_ID')
for i in colnames[2:]:
    values=np.full(14,'NaN',dtype='object')
    new_tab.add_column(values,name=i)
    
new_tab.write(table_path,format='ascii',overwrite=True)
'''

#Adding and removing
#new_col = Column(name='Gaia_ID',data=[[1],[1],[1],[1],[1],[1],[1],
#                                      [1],[1],[1],[1],[1],[1],[1]])
#data.replace_column('Gaia_ID',new_col)
#data.remove_column('d')

#dat =Table.read('test.dat',format='ascii')
#print(dat)

#print(type(data['Gaia_ID'].data))
#data.write(table_path,format='ascii',overwrite=True)

