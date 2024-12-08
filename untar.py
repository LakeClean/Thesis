import zipfile
import glob
import os
import astropy.io.fits as pyfits
import tarfile
import shutil


'''
Script for unzipping the files from HARPS and taking the
ordered files and putting them in the right directory.
'''

'''
#HARPS:
path = '/home/lakeclean/Downloads/'


filenames = ['KIC10454113.tar.gz', 'KIC9025370.tar.gz']
taredfiles = []
for file in filenames:
    taredfiles.append(glob.glob(f'{path}{file}')[0])

print(taredfiles)
temporary_dir = f"{path}temp_dir"

try:
    os.mkdir(temporary_dir)
except:
    files = glob.glob(temporary_dir + '/*')
    for file in files:
        os.remove(file)
    os.rmdir(temporary_dir)
    os.mkdir(temporary_dir)


for taredfile in taredfiles:
    file = tarfile.open(taredfile)
    file.extractall(temporary_dir)


files = glob.glob(temporary_dir + '/*')

print(files)
final_dir = '/home/lakeclean/Documents/speciale/initial_data/TNG/'



for file in files:
    directories = glob.glob(f'{final_dir}*')
    header = pyfits.getheader(file)
    epoch_date = header['DATE-OBS'].strip(' ')
    date_begin = epoch_date[0:10]
    print(epoch_date,date_begin)
    
    if final_dir + date_begin in directories:
        new_name = final_dir + date_begin + file[len(temporary_dir):]
        os.rename(file,new_name)

    else:
        os.mkdir(final_dir + date_begin)
        new_name = final_dir + date_begin + file[len(temporary_dir):]
        os.rename(file,new_name)

        


files = glob.glob(temporary_dir + '/*')
for file in files:
        os.remove(file)
os.rmdir(temporary_dir)
'''

#HIRES:
path = '/home/lakeclean/Downloads/'


folders = ['KOA_69614']

files = []
for folder in folders:
    print(folder)
    for i in range(3):
        file = glob.glob(path + f'{folder}/HIRES/extracted/binaryfits/ccd{1+i}/flux/*')
        print(path + f'{folder}/HIRES/extracted/binaryfits/ccd{1+i}/flux/*')
        for f in file:
            files.append(f)

print(files)

final_dir = '/home/lakeclean/Documents/speciale/initial_data/KECK/KIC10454113/'

length = len('/home/lakeclean/Downloads/KOA_66711/HIRES/extracted/binaryfits/ccd1/flux')

for file in files:
    directories = glob.glob(f'{final_dir}*')
    header = pyfits.getheader(file)
    epoch_date = header['DATE-OBS'].strip(' ')
    date_begin = epoch_date
    print(epoch_date,header['TARGNAME'])
    new_name = final_dir + date_begin + file[length:]
    
    

    if final_dir + date_begin in directories:
        new_name = final_dir + date_begin + file[length:]
        shutil.copyfile(file,new_name)

    else:
        os.mkdir(final_dir + date_begin)
        new_name = final_dir + date_begin + file[length:]
        shutil.copyfile(file,new_name)
















