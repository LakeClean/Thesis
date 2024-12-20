import zipfile
import glob
import os
import astropy.io.fits as pyfits
import tarfile
import shutil


'''
Script for unzipping the files from NOT and taking the ordered files and
putting them in the right directory. Can be used for new files as well.
Probably....

'''

#NOT:

path = '/home/lakeclean/Downloads/'

zipedfiles = glob.glob(f'{path}reduced*')


temporary_dir = f"{path}temp_dir"

try:
    os.mkdir(temporary_dir)
except:
    files = glob.glob(temporary_dir + '/*')
    for file in files:
        os.remove(file)
    os.rmdir(temporary_dir)
    os.mkdir(temporary_dir)


for zipedfile in zipedfiles:
    with zipfile.ZipFile(zipedfile,"r") as zip_ref:
        zip_ref.extractall(temporary_dir)


files = glob.glob(temporary_dir + '/*')

final_dir = '/home/lakeclean/Documents/speciale/initial_data/NOT/'



for file in files:
    directories = glob.glob(f'{final_dir}*')
    header = pyfits.getheader(file)
    epoch_date = header['DATE-OBS'].strip(' ')
    date_begin = epoch_date[0:10]
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
#When files are not zipped
path = '/home/lakeclean/Documents/speciale/initial_data/NOT_old/'

files = glob.glob(f'{path}*wave.fits')


final_dir = '/home/lakeclean/Documents/speciale/initial_data/NOT/'



for file in files:
    directories = glob.glob(f'{final_dir}*')
    header = pyfits.getheader(file)
    epoch_date = header['DATE-OBS'].strip(' ')
    date_begin = epoch_date[0:10]
    if final_dir + date_begin in directories:
        new_name = final_dir + date_begin + file[len(path)-1:]
        shutil.copyfile(file,new_name)

    else:
        os.mkdir(final_dir + date_begin)
        new_name = final_dir + date_begin + file[len(path)-1:]
        print(new_name)
        shutil.copyfile(file,new_name)
        
'''
















