import numpy as np
import astropy.io.fits as pyfits
import glob
import barycorrpy as bc
from astropy.time import Time
import make_table_of_target_info as mt
master_path = '/usr/users/au662080'



'''
directory = f'{master_path}/Speciale/data/initial_data'
folders = glob.glob(f'{directory}/*')
f = open(f'{master_path}/Speciale/data/merged_file_log.txt','w')
f.write("ID, SEQID, directory, date, T_exp, v_helio, fiber,\
                        npixels, CDELT1, CRVAL1, TELESCOP \n")
info = []
for folder in folders:

    files = glob.glob(f'{folder}/*merge.fits')
    for file in files:

        header = pyfits.getheader(file)
        ID = header['TCSTGT'].replace('-','') #The ID of the target
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

'''
#Sorting based on date:
def mySort(x):
    x = x.split(',')
    return x[2]

tab = mt.get_table()
all_target_names = tab['ID'].data # Just a list of the names
all_ras = tab['RA'].data # right ascension of all stars
all_decs = tab['DEC'].data #declination of all stars


pm_dir = f'{master_path}/Speciale/data/propermotion_parallax.txt'
pm_lines = open(pm_dir).read().split('\n')


directory = f'{master_path}/Speciale/data/initial_data/NOT'
folders = glob.glob(f'{directory}/*')

f = open(f'{master_path}/Speciale/data/NOT_order_file_log.txt','w')
head = "ID, directory, date,T_exp,v_helio,fiber,npixels"
head = head + ",ra,dec,lat,longi,alt,epoch,pmra,pmdec,px"
head = head + ",epoch_jd,v_bary,TELESCOP"
f.write(head+"\n")
info = []
k=0
for folder in folders:

    files = glob.glob(f'{folder}/*10_wave.fits')
    for file in files:
        #print(file)
        header = pyfits.getheader(file)
        
        try:
            SEQID = header['SEQID'] #either science or thorium argon
        except:
            SEQID = 'science'

        if SEQID != 'science':
                print('removed:', SEQID, k)
                k+=1
                continue
        
        ra = header['RA'] #The right ascension of target
        dec = header['DEC'] #The declination of target
        #Finding out the name of target based on RA and DEC:
        coord_dist = np.sqrt( (all_ras- ra)**2 + (all_decs-dec)**2)
        ID = all_target_names[np.where(coord_dist == min(coord_dist))[0]][0]
        #ID = header['TCSTGT'].replace('-','') #The ID of the target

        try: #Specific test for old not targets
            test = int(ID[0])
            if test == 0:
                ID = 'KIC' + ID[1:]
            else:
                ID = 'KIC' + ID
        except:
            pass

        
        T_exp = header['EXPTIME'] #exposure time
        npixels = header['NAXIS1'] #number of pixels wavelengthwise
        fiber = header['FIFMSKNM'] #The type of fiber, 4=high res
        v_helio = header['VHELIO'] #heliocentric velocity
        date = header['DATE-OBS'] #Fitsfile creation date
        TELESCOP = header['TELESCOP'] #Telescope used.
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
        data = f'{ID},{file},{date},{T_exp},{v_helio},{fiber},'
        data = data + f'{npixels},{ra},{dec},{lat},{longi},'
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
        
        

info.sort(key = mySort)  

for i in info:
    f.write(i)

f.close()

'''

#Sorting based on date:
def mySort(x):
    x = x.split(',')
    return x[2]

tab = mt.get_table()
all_target_names = tab['ID'].data # Just a list of the names
all_ras = tab['RA'].data # right ascension of all stars
all_decs = tab['DEC'].data #declination of all stars


pm_dir = f'{master_path}/Speciale/data/propermotion_parallax.txt'
pm_lines = open(pm_dir).read().split('\n')

directory = f'{master_path}/Speciale/data/initial_data/NOT'
folders = glob.glob(f'{directory}/*')


f_new = open(f'{master_path}/Speciale/data/NOT_order_file_log.txt','w')
f_old_LOWRES = open(f'{master_path}/Speciale/data/NOT_old_LOWRES_order_file_log.txt','w')
f_old_HIRES = open(f'{master_path}/Speciale/data/NOT_old_HIRES_order_file_log.txt','w')
head = "ID,directory,date,T_exp,v_helio,fiber,npixels"
head = head + ",ra,dec,lat,longi,alt,pmra,pmdec,px"
head = head + ",epoch_jd,v_bary,TELESCOP"
f_new.write(head+"\n")
f_old_LOWRES.write(head+"\n")
f_old_HIRES.write(head+"\n")
info_new = []
info_old_LOWRES = []
info_old_HIRES = []
k=0
for folder in folders:

    files = glob.glob(f'{folder}/*10_wave.fits')
    for file in files:
        header = pyfits.getheader(file)
        
        try:
            SEQID = header['SEQID'] #either science or thorium argon
        except:
            SEQID = 'science' #The old obs don't have SEQID

        if SEQID != 'science':
                print('removed:', SEQID, k)
                k+=1
                continue
        
        ra = header['RA'] #The right ascension of target
        dec = header['DEC'] #The declination of target
        #Finding out the name of target based on RA and DEC:
        coord_dist = np.sqrt( (all_ras- ra)**2 + (all_decs-dec)**2)
        ID = all_target_names[np.where(coord_dist == min(coord_dist))[0]][0]
        #ID = header['TCSTGT'].replace('-','') #The ID of the target

        try: #Specific test for old not targets
            test = int(ID[0])
            if test == 0:
                ID = 'KIC' + ID[1:]
            else:
                ID = 'KIC' + ID
        except:
            pass

        
        T_exp = header['EXPTIME'] #exposure time
        npixels = header['NAXIS1'] #number of pixels wavelengthwise
        fiber = header['FIFMSKNM'] #The type of fiber, 4=high res
        v_helio = header['VHELIO'] #heliocentric velocity
        date = header['DATE-OBS'] #Fitsfile creation date
        TELESCOP = header['TELESCOP'] #Telescope used.
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
        data = f'{ID},{file},{date},{T_exp},{v_helio},{fiber},'
        data = data + f'{npixels},{ra},{dec},{lat},{longi},'
        data = data + f'{alt},{pmra},{pmdec},{px},{epoch_jd},{v_bary},'
        data = data + f'{TELESCOP}'


        if epoch_jd < 2457506: #The newest old NOT file
            if fiber == 'F1 LowRes':
                info_old_LOWRES.append(data + '\n')
            if fiber == 'F4 HiRes':
                info_old_HIRES.append(data + '\n')
        if epoch_jd > 2457506: #The newest old NOT file
            info_new.append(data + '\n')
        
        

info_new.sort(key = mySort)
info_old_LOWRES.sort(key = mySort)
info_old_HIRES.sort(key = mySort)

for i in info_new:
    f_new.write(i)

f_new.close()

for i in info_old_LOWRES:
    f_old_LOWRES.write(i)

f_old_LOWRES.close()

for i in info_old_HIRES:
    f_old_HIRES.write(i)

f_old_HIRES.close()




















      
        
