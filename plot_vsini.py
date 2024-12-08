import numpy as np
import pandas as pd
import make_table_of_target_info as mt
import glob
import matplotlib.pyplot as plt

IDs = mt.get_table()['ID'].data

'''
path = '/home/lakeclean/Documents/speciale/target_analysis/'

for ID in IDs[0:1]:
    print(ID)
    dates = glob.glob(path+f'{ID}/*')
    for date in dates[0:]:
        #fig, ax = plt.subplots()
        print(date)
        df = pd.read_csv(date + '/data/bf_fit_params.txt')
        print(df['epoch_vsini1'].to_numpy()[30:60])
        
'''

path = '/home/lakeclean/Documents/speciale/rv_data'

for ID in IDs:
    df = pd.read_csv(path + f'/rv_{ID}.txt')

    #print(df['vsini1'])



    fig, ax  = plt.subplots()
    ax.set_title(f'ID: {ID}')
    ax.scatter(df['jd'].to_numpy()- 2457000,df['vsini1'].to_numpy())
    ax.set_ylabel('vsini km/s')
    ax.set_xlabel('JD - 2457000[days]')

    plt.show()
    






