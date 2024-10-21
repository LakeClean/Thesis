import glob
import pandas as pd

names = []

lines = open('/home/lakeclean/Documents/speciale/order_file_log.txt').read().split('\n')
files = []
IDs = []
dates = []
for line in lines[:-1]:
    line = line.split(',')
    file = line[2].strip()
    SEQID = line[1].strip()
    ID = line[0].strip()
    date = line[3].strip()
    if SEQID == 'science':
        files.append(file)
        IDs.append(ID)
        dates.append(date)
        if ID not in names:
            names.append(ID)
run=False #WARNING DONT RUN THIS CODE! IT WILL REMOVE NOTES!

if run:
    f  =open('/home/lakeclean/Documents/speciale/spectra_log_h_readable.txt','w')
    for name in names:
        
        nr = 0
        out = ''
        for date, ID in zip(dates,IDs):
            if ID == name:
             out += f'  {date},         ,          ,           , \n'
             nr += 1
             
             

        f.write(f'########## {name}, nr. obs: {nr} ###########\n')
        f.write('  date,'.center(25) + 'SB type,'.center(15) + 'V_rad,'.center(11) + 'Vsini,'.center(11) +'\n')
        f.write(f'{out}')
        f.write('&')

    f.close()






















         
            
