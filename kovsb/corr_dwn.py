import sys
import numpy as np
import pandas as pd
from nsstools import NssSource

star_names = ("KIC9693187", "KIC9025370", "KIC4914923", "KIC12317678")
#("t_periastron", "a0", "eccentricity", "inclination", "arg_periastron", "nodeangle", "period", "parallax")
param_names = ("t_periastron", "a0", "eccentricity", "inclination", "arg_periastron", "nodeangle", "period", "parallax")
gaia_fit_names = ("parallax", "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes", "eccentricity", "period", "t_periastron")
drop_i = ["ra", "dec", "pmra", "pmdec", "center_of_mass_velocity", "c_thiele_innes", "h_thiele_innes"]
drop_j = ["ra", "dec", "pmra", "pmdec"]

#nss = pd.read_csv("/usr/users/au649504/kovsb/Jonatan.csv")
files = pd.read_csv("Jonatan.csv")
nss = pd.read_csv("Jonatan.csv")

source_index = 1 #2
source = NssSource(nss, indice = source_index)
covmat = source.covmat()

for n, i in enumerate(star_names):
    source_index = n # position of the source in the csv file
    cvs_file = files[n:n + 1]
    source = NssSource(nss, indice = source_index)
    try:
        covmat = source.covmat().drop(columns = drop_i).drop(drop_i)
    except KeyError:
        covmat = source.covmat().drop(columns = drop_j).drop(drop_j)
    covmat_np = covmat.to_numpy()
    
    campbell = source.campbell()
    
    params = np.array([])
    for m, j in enumerate(param_names):
        if m not in [0, 2, 6, 7]:
            params = np.append(params, campbell[j].values[0])
        else:
            params = np.append(params, cvs_file[j].values[0])
    
    gaia_ti = np.array([])
    for k in gaia_fit_names:
        gaia_ti = np.append(gaia_ti, cvs_file[k].values[0])
        
    np.save(f"{i}_gaia_par.npy", params)
    np.save(f"{i}_gaia_ti.npy", gaia_ti)
    np.save(f"{i}_gaia_cmat.npy", covmat_np)
