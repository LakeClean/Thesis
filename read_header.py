import numpy as np
import astropy.io.fits as pyfits
import glob

directory = '/home/lakeclean/Documents/speciale/initial_data'
folders = glob.glob(f'{directory}/*')
f = open('file_log.txt','w')
f.write("ID, SEQID, directory, date, T_exp, v_helio, fiber,\
                        npixels, CDELT1, CRVAL1 \n")
info = []
for folder in folders:

    files = glob.glob(f'{folder}/*merge.fits')
    for file in files:

        header = pyfits.getheader(file)
        ID = header['TCSTGT'] #The ID of the target
        T_exp = header['EXPTIME'] #exposure time
        npixels = header['NAXIS1'] #number of pixels wavelengthwise
        fiber = header['FIFMSKNM'] #The type of fiber, 4=high res
        v_helio = header['VHELIO'] #heliocentric velocity
        CDELT1 = header['CDELT1'] #wavelength per pixel
        CRVAL1 = header['CRVAL1'] #Starting wavelength
        date = header['DATE'] #Fitsfile creation date
        SEQID = header['SEQID'] #either science or thorium argon
        
        info.append(f'{ID}, {SEQID}, {file}, {date}, {T_exp}, {v_helio}, {fiber},\
                {npixels}, {CDELT1}, {CRVAL1} \n')

#Sorting based on date:
def mySort(x):
    x = x.split(',')
    return x[3]

info.sort(key = mySort)  

for i in info:
    f.write(i)

f.close()


        
        
