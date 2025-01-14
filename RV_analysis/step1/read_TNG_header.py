import numpy as np
import astropy.io.fits as pyfits
import glob
import barycorrpy as bc
from astropy.time import Time
import make_table_of_target_info as mt
master_path = '/usr/users/au662080'



'''
directory = 'f'{master_path/Speciale/data/initial_data'
folders = glob.glob(f'{directory}/*')
f = open('f'{master_path/Speciale/data/merged_file_log.txt','w')
f.write("ID, SEQID, directory, date, T_exp, v_helio, fiber,\
                        npixels, CDELT1, CRVAL1, TELESCOP \n")
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
        TELESCOP = header['TELESCOP'] #Telescope used.
        
        info.append(f'{ID}, {SEQID}, {file}, {date}, {T_exp}, {v_helio}, {fiber},\
                {npixels}, {CDELT1}, {CRVAL1}, {TELESCOP} \n')


info.sort(key = mySort)  

for i in info:
    f.write(i)

f.close()
'''

#Sorting based on date:
def mySort(x):
    x = x.split(',')
    return x[3]

pm_dir = f'{master_path}/Speciale/data/propermotion_parallax.txt'
pm_lines = open(pm_dir).read().split('\n')


directory = f'{master_path}/Speciale/data/initial_data/TNG'
folders = glob.glob(f'{directory}/*')
f = open(f'{master_path}/Speciale/data/TNG_merged_file_log.txt','w')
head = "ID,directory,date,T_exp,npixels,CDELT1,CRVAL1"
head = head + ",ra,dec,lat,longi,alt,pmra,pmdec,px"
head = head + ",epoch_jd,v_bary,TELESCOP"
f.write(head+"\n")
info = []

for folder in folders:

    files = glob.glob(f'{folder}/HARPN*.fits') #importing unaltered files
    for file in files:
        print(file)
        
        header = pyfits.getheader(file)
        ID = header['HIERARCH TNG OBS TARG NAME'] #The ID of the target
        T_exp = header['EXPTIME'] #exposure time
        npixels = header['NAXIS1'] #number of pixels wavelengthwise
        CDELT1 = header['CDELT1'] #wavelength per pixel
        CRVAL1 = header['CRVAL1'] #Starting wavelength
        #fiber = header['FIFMSKNM'] #The type of fiber, 4=high res
        #v_helio = header['VHELIO'] #heliocentric velocity
        date = header['DATE-OBS'] #Fitsfile creation date
        #SEQID = header['OBS-TYPE'] #either science or thorium argon
        TELESCOP = header['TELESCOP'] #Telescope used.
        ra = header['RA-DEG'] #The right ascension of target in degrees
        dec = header['DEC-DEG'] #The declination of target in degrees
        lat = 28.75 # latitude of instrument
        longi = -17.89 #longitude of instrument
        alt = 2370 #altitude of instrument in meters
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
        data = f'{ID},{file},{date},{T_exp},{npixels},'
        data = data + f'{CDELT1},{CRVAL1},{ra},{dec},{lat},{longi},'
        data = data + f'{alt},{pmra},{pmdec},{px},{epoch_jd},{v_bary},'
        data = data + f'{TELESCOP}'
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






















      
        
