# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:18:28 2024

@author: Jonatan Rudrasingam
"""

import sys
import numpy as np


time_rv, rva, rvb, rva_err, rvb_err, instrument = np.load("rv_sb2.npy", allow_pickle = True)
time_ast, rho, theta, rho_err, theta_err = np.load("ast_pa_sep5.npy", allow_pickle = True)
# Calculate the correlation coefficient between theta and rho
c = np.corrcoef((rho, theta))[0][1]

def write_sb1_txt(npy_file: str, outname: str):
    """
    Creates a txt file containing time, RV, RV error and instrument ID from
    a npy file.
        
    :params:
      npy.file      : str, name of the npy file
      outname       : str, the name of the output txt file
    
    """
    
    time_rv, rva, rva_err, instrument = np.load(npy_file, allow_pickle = True)
    instrument = instrument.astype(int)
    file = open(outname, "w")
    file.write(("{:15s} {:>13s} {:>24s} {:>22s}".format("Epoch [D]", "RV [km/s]", "RV_err [km/s]", "InstrumentID")))
    file.write('\n' )
    for i in np.arange(len(time_rv)):
        file.write(("{:<15.8f} {:>15.8f} {:>19.8f} {:>14}".format(time_rv[i], rva[i], rva_err[i], instrument[i])))
        file.write('\n' )
    file.close()


def write_sb2_txt(npy_file: str, outname1: str, outname2: str):
    """
    Creates tow txt files containing time, RV, RV error and instrument ID for
    both conponents from a npy file.
        
    :params:
      npy.file      : str, name of the npy file
      outname1      : str, the name of the output txt file containing RV from A
      outname2      : str, the name of the output txt file containing RV from B
    
    """
    
    time_rv, rva, rvb, rva_err, rvb_err, instrument = np.load(npy_file, allow_pickle = True)
    instrument = instrument.astype(int)
    
    file = open(outname1, "w")
    file.write(("{:15s} {:>13s} {:>24s} {:>22s}".format("Epoch [D]", "RV [km/s]", "RV_err [km/s]", "InstrumentID")))
    file.write('\n' )
    for i in np.arange(len(time_rv)):
        file.write(("{:<15.8f} {:>15.8f} {:>19.8f} {:>14}".format(time_rv[i], rva[i], rva_err[i], instrument[i])))
        file.write('\n' )
    file.close()
    
    file = open(outname2, "w")
    file.write(("{:15s} {:>13s} {:>24s} {:>22s}".format("Epoch [D]", "RV [km/s]", "RV_err [km/s]", "InstrumentID")))
    file.write('\n' )
    for i in np.arange(len(time_rv)):
        file.write(("{:<15.8f} {:>15.8f} {:>19.8f} {:>14}".format(time_rv[i], rvb[i], rvb_err[i], instrument[i])))
        file.write('\n' )
    file.close()


def write_sb2_b_txt(npy_file: str, outname1: str, outname2: str):
    """
    Another function that creates tow txt files containing time, RV, RV error and 
    instrument ID for both conponents from a npy file.
        
    :params:
      npy.file      : str, name of the npy file
      outname1      : str, the name of the output txt file containing RV from A
      outname2      : str, the name of the output txt file containing RV from B
    
    """
    
    time_rva, time_rvb, rva, rva_err, rvb, rvb_err, instrument_a, instrument_b = np.load(npy_file, allow_pickle = True).T
    instrument_a = instrument_a.astype(int)
    instrument_b = instrument_b.astype(int)
    
    file = open(outname1, "w")
    file.write(("{:15s} {:>13s} {:>24s} {:>22s}".format("Epoch [D]", "RV [km/s]", "RV_err [km/s]", "InstrumentID")))
    file.write('\n' )
    for i in np.arange(len(time_rva)):
        file.write(("{:<15.8f} {:>15.8f} {:>19.8f} {:>14}".format(time_rva[i], rva[i], rva_err[i], instrument_a[i])))
        file.write('\n' )
    file.close()
    
    file = open(outname2, "w")
    file.write(("{:15s} {:>13s} {:>24s} {:>22s}".format("Epoch [D]", "RV [km/s]", "RV_err [km/s]", "InstrumentID")))
    file.write('\n' )
    for i in np.arange(len(time_rvb)):
        file.write(("{:<15.8f} {:>15.8f} {:>19.8f} {:>14}".format(time_rvb[i], rvb[i], rvb_err[i], instrument_b[i])))
        file.write('\n' )
    file.close()
    
    
def write_rel_ast_txt(npy_file: str, outname: str):
    """
    Creates a txt file containing time, RV, RV error and instrument ID from
    a npy file.
        
    :params:
      npy.file      : str, name of the npy file
      outname       : str, the name of the output txt file
    
    """
    
    time_ast, rho, theta, rho_err, theta_err = np.load(npy_file, allow_pickle = True)
    file = open(outname, "w")
    file.write((("{:15s} {:>14s} {:>23s} {:>16s} {:>24s}".format("Epoch [D]", "Sep ['']", "Sep_err ['']", "PA []", "PA_err [°]"))))
    file.write('\n' )
    for i in np.arange(len(time_ast)):
        file.write(("{:<15.8f} {:>15.8f} {:>19.8f} {:>19.4f} {:>19.4f}".format(time_ast[i], rho[i], rho_err[i], theta[i], theta_err[i],)))
        file.write('\n' )
    file.close()

def write_gaia_ti_txt(npy_file: str, outname: str):
    """
    Creates a txt file containing Gaia Thiele Innes elements froma npy file.
        
    :params:
      npy.file      : str, name of the npy file
      outname       : str, the name of the output txt file
    
    """
    
    pi, A, B, F, G, e, p, t0 = np.load(npy_file, allow_pickle = True)
    file = open(outname, "w")
    file.write(("{:15s} {:>14s} {:>20s} {:>18s} {:>24s} {:>24s} {:>17s} {:>24s}".format("parallax", "A_Thiele_Innes", "B_Thiele_Innes", "F_Thiele_Innes", "G_Thiele_Innes", "eccentricity", "period", "t_periastron")))
    file.write('\n' )
    file.write(("{:<15f} {:>8f} {:>20f} {:>18f} {:>24f} {:>26f} {:>24f} {:>19f}".format(pi, A, B, F, G, e, p, t0)))
    file.close()
    

datadir = "C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\"
#write_sb1_txt(datadir + "rv_sb1.npy", datadir + "rv_chi_dra_a.txt")
#write_sb2_txt(datadir + "rv_sb2.npy", datadir + "rv_chi_dra_a.txt", datadir + "rv_chi_dra_b.txt")
#write_rel_ast_txt(datadir + "ast_pa_sep5.npy", datadir + "rel_ast_chi_dra.txt")
#write_sb2_b_txt(datadir + "jonatan_KIC9025370.npy", datadir + "rv_kic9025370_a.txt", datadir + "rv_kic9025370_b.txt")

time_rv, rva, rva_err, instrument = np.loadtxt("C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\rv_chi_dra_a.txt", skiprows = 1).T
time_rv, rvb, rvb_err, instrument = np.loadtxt("C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\rv_chi_dra_b.txt", skiprows = 1).T
time_ast, rho, rho_err, theta, theta_err = np.loadtxt("C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\rel_ast_chi_dra.txt", skiprows = 1).T
# Calculate the correlation coefficient between theta and rho
c = np.corrcoef((rho, theta))[0][1]

t0, a0, e, i, w, W, p, pi = np.load("KIC9025370_gaia_par.npy", allow_pickle = True)

write_gaia_ti_txt("KIC9025370_gaia_ti.npy", "C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\kic9025370_gti.txt")
write_gaia_ti_txt("KIC4914923_gaia_ti.npy", "C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\kic4914923_gti.txt")
write_gaia_ti_txt("KIC9693187_gaia_ti.npy", "C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\kic9693187_gti.txt")
write_gaia_ti_txt("KIC12317678_gaia_ti.npy", "C:\\Users\\Jonatan\\Documents\\Jonatan\\kovsb\\data\\kic12317678_gti.txt")
