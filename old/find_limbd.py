import numpy as np

'''
Simple script for estimating the limbddarkening coefficient of function
for broadening function from (Kaluzny, 2006).
The coefficients come from the analysis (van Hamme, 1993)
'''

path = '/home/lakeclean/Documents/speciale/limcof_rucinski_p00.dat.txt'

def find_limbd(logg,Teff):
    '''
    We simply try to find the limbd that is closest relatively to
    the parameters describing it of logg and Teff [K]
    '''
    
    Teffs = np.arange(3500,10000,250)
    loggs = np.arange(0,5.5,0.5)
    
    d_Teffs = abs(1 - np.arange(3500,10000,250)/Teff)
    d_loggs = abs(1 - np.arange(0,5.5,0.5)/logg)

    T_index = np.where(d_Teffs == min(d_Teffs))[0]
    logg_index = np.where(d_loggs ==min(d_loggs))[0]

    Teff,logg = Teffs[T_index],loggs[logg_index]

    f = open(path).read().split('\n')

     
    
    


    

find_limbd(2.8,4800)
