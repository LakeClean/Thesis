import numpy as np
import pandas as pd

master_path = '/home/lakeclean/Documents/speciale/table_of_info.txt'

df = pd.read_csv(master_path,sep='|',dtype=object)
nr_IDs = np.shape(df)[0]

def add_key(key):
    '''
    input:
        - key: [str] no spaces around it!
    '''
    for i in df.keys():
        if key == i:
            print(f'The key: {key} already exist')
            return
            
    df[key] = np.empty(nr_IDs,dtype=object)
    df.to_csv(master_path,sep='|', index=False)
    

def get_value(key,ID):
    for i,j in enumerate(df['ID']):
        if j == ID:
            index = i
    return df[key][index]
    

def add_value(key,ID,value):
    '''
    input:
        - key: [str] no spaces around it!
        - ID: [str] The ID of the target you want to add info to
        - value: [list] The list of values you want [values,errors,sources]
                - values: list of values
                - errors: list of errors for the above values
                - sources: list of sources for the above
                
    '''
    for i,j in enumerate(df['ID']):
        if j == ID:
            index = i

    df.loc[index, key] = f'{value}'
    df.to_csv(master_path,sep='|',index=False)
    

def get_table():
    return df
    




