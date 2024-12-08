import numpy as np
import astropy.io.fits as pyfits
import glob
import barycorrpy as bc
from astropy.time import Time
import make_table_of_target_info as mt
from astropy import units as u
from astropy.coordinates import SkyCoord


#Sorting based on date:
def mySort(x):
    x = x.split(',')
    return x[3]

pm_dir = '/home/lakeclean/Documents/speciale/propermotion_parallax.txt'
pm_lines = open(pm_dir).read().split('\n')


target_ID = 'KIC10454113'

directory = f'/home/lakeclean/Documents/speciale/initial_data/KECK/{target_ID}'
folders = glob.glob(f'{directory}/*')
f = open('/home/lakeclean/Documents/speciale/KECK_ordered_file_log.txt','w')
head = "ID, directory, date, T_exp, HELIOVEL, "
head = head + "ra, dec, lat, longi, alt, pmra, pmdec, px"
head = head + ", epoch_jd, v_bary, TELESCOP"
f.write(head+"\n")
info = []

for folder in folders:

    files = glob.glob(f'{folder}/HI*') #importing unaltered files

    for file in files:
        #print(file)
        
        header = pyfits.getheader(file)
        ID = target_ID #The ID of the target
        T_exp = header['EXPTIME'] #exposure time
        
        #npixels = header['NAXIS1'] #number of pixels wavelengthwise
        #CDELT1 = header['CDELT1'] #wavelength per pixel
        #CRVAL1 = header['CRVAL1'] #Starting wavelength
        #fiber = header['FIFMSKNM'] #The type of fiber, 4=high res
        VHELIO = header['HELIOVEL'] #heliocentric velocity
        date = header['DATE'] #Fitsfile creation date
        #SEQID = header['OBS-TYPE'] #either science or thorium argon
        TELESCOP = 'KECK' #Telescope used.
        c = SkyCoord(header['RA'],header['DEC'],unit=(u.hourangle, u.deg))
        ra = np.float64(c.ra) #The right ascension of target in degrees
        dec = np.float64(c.dec) #The declination of target in degrees
        lat = 19.8263 # latitude of instrument
        longi = -155.47441 #longitude of instrument
        alt = 4145 #altitude of instrument in meters
        #Parallax and pm in mas
        px = 0
        pmra = 0
        pmdec = 0
        e_px = 0
        e_pmra = 0
        e_pmdec = 0
        for line in pm_lines:
            line = line.split(',')
            if line[0].strip(' ') == ID.strip(' '):
                pmra = float(line[1].strip(' '))
                pmdec = float(line[2].strip(' '))
                px = float(line[5].strip(' '))
                e_pmra = float(line[3].strip(' '))
                e_pmdec = float(line[4].strip(' '))
                e_px = float(line[6].strip(' '))
        # epoch in JD
        epoch_jd = Time(date.strip(' ')).jd #julean date

        #using barycorrpy
        result = bc.get_BC_vel(JDUTC=epoch_jd, ra=ra, dec=dec,
                               lat=lat, longi=longi, alt=alt,pmra=pmra,
                               pmdec=pmdec, px=px, leap_update=True)
        
        v_bary = result[0][0] # barycentric velocity in m/s
        data = f'{ID}, {file}, {date}, {T_exp}, {VHELIO}, '
        data = data + f'{ra}, {dec}, {lat}, {longi}, '
        data = data + f'{alt}, {pmra}, {pmdec}, {px}, {epoch_jd}, {v_bary}, {TELESCOP}'
        info.append(data + '\n')
        
        
    

info.sort(key = mySort)  

for i in info:
    f.write(i)

f.close()






















      
        
