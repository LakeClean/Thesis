import numpy as np
import astropy.io.fits as pyfits
import glob
import barycorrpy as bc
from astropy.time import Time
import make_table_of_target_info as mt


directory = '/home/lakeclean/Documents/speciale/initial_data'
folders = glob.glob(f'{directory}/*')
f = open('/home/lakeclean/Documents/speciale/merged_file_log.txt','w')
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
        date = header['DATE_OBS'] #Date of observation
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

pm_dir = '/home/lakeclean/Documents/speciale/propermotion_parallax.txt'
pm_lines = open(pm_dir).read().split('\n')


directory = '/home/lakeclean/Documents/speciale/initial_data'
folders = glob.glob(f'{directory}/*')
f = open('/home/lakeclean/Documents/speciale/order_file_log.txt','w')
head = "ID, SEQID, directory, date, T_exp, v_helio, fiber,npixels"
head = head + ", ra, dec, lat, longi, alt, epoch, pmra, pmdec, px"
head = head + ", epoch_jd, v_bary"
f.write(head+"\n")
info = []

for folder in folders:

    files = glob.glob(f'{folder}/*wave.fits')
    for file in files:
        
        header = pyfits.getheader(file)
        ID = header['TCSTGT'] #The ID of the target
        T_exp = header['EXPTIME'] #exposure time
        npixels = header['NAXIS1'] #number of pixels wavelengthwise
        fiber = header['FIFMSKNM'] #The type of fiber, 4=high res
        v_helio = header['VHELIO'] #heliocentric velocity
        date = header['DATE_OBS'] #Fitsfile creation date
        SEQID = header['SEQID'] #either science or thorium argon
        ra = header['RA'] #The right ascension of target
        dec = header['DEC'] #The declination of target
        lat = 28.757222222 # latitude of instrument
        longi = -17.885 #longitude of instrument
        alt = 2465.5 #altitude of instrument in meters
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
        data = f'{ID}, {SEQID}, {file}, {date}, {T_exp}, {v_helio}, {fiber}, '
        data = data + f'{npixels}, {ra}, {dec}, {lat}, {longi}, '
        data = data + f'{alt}, {pmra}, {pmdec}, {px}, {epoch_jd}, {v_bary}'
        info.append(data + '\n')

        # Writing to master_table:
        #mt.add_value(pmra,'e_pmra',ID)
        #mt.add_value(pmdec,'e_pmdec',ID)
        #mt.add_value(px,'e_px',ID)
        #mt.add_value('SIMBAD[mas]','px_source',ID)
        #mt.add_value('SIMBAD[mas]','pmra_source',ID)
        #mt.add_value('SIMBAD[mas]','pmdec_source',ID)
        #print( ID == mt.get_table()['ID'])
        #print(ID,pmra)
        
        
        
        

#Sorting based on date:
def mySort(x):
    x = x.split(',')
    return x[3]

info.sort(key = mySort)  

for i in info:
    f.write(i)

f.close()






















      
        
