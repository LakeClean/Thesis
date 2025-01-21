import glob
import pandas as pd
import make_table_of_target_info as mt
names = []
files = []
IDs = []
dates = []

master_path = '/usr/users/au662080'

#New NOT
lines = open(f'{master_path}/Speciale/data/NOT_order_file_log.txt').read().split('\n')
for line in lines[:-1]:
    line = line.split(',')
    file = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)
    if ID not in names:
        names.append(ID)

#old NOT HIRES
lines = open(f'{master_path}/Speciale/data/NOT_old_HIRES_order_file_log.txt').read().split('\n')
for line in lines[:-1]:
    line = line.split(',')
    file = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)
    if ID not in names:
        names.append(ID)

#old NOT LOWRES
lines = open(f'{master_path}/Speciale/data/NOT_old_LOWRES_order_file_log.txt').read().split('\n')
for line in lines[:-1]:
    line = line.split(',')
    file = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)
    if ID not in names:
        names.append(ID)

#TNG
lines = open(f'{master_path}/Speciale/data/TNG_merged_file_log.txt').read().split('\n')
for line in lines[:-1]:
    line = line.split(',')
    file = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)
    if ID not in names:
        names.append(ID)


#KECK
lines = open(f'{master_path}/Speciale/data/KECK_order_file_log.txt').read().split('\n')
for line in lines[:-1]:
    line = line.split(',')
    file = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)
    if ID not in names:
        names.append(ID)


#ESpaDOns
lines = open(f'{master_path}/Speciale/data/ESpaDOns_merged_file_log.txt').read().split('\n')
for line in lines[:-1]:
    line = line.split(',')
    file = line[1].strip()
    ID = line[0].strip()
    date = line[2].strip()
    files.append(file)
    IDs.append(ID)
    dates.append(date)
    if ID not in names:
        names.append(ID)



run=False #WARNING DONT RUN THIS CODE! IT WILL REMOVE NOTES!
'''
SB2IDs =['KIC-9693187','KIC-9025370','KIC9652971']
IDlines = open(f'{master_path}/Speciale/data/spectra_log_h_readable.txt').read().split('&')
SB2_IDs, SB2_dates, SB2_types, vguess1s, vguess2s = [], [], [], [], []
for IDline in IDlines[:-1]:
    if IDline.split(',')[0][11:].strip(' ') in SB2IDs:
        for line in IDline.split('\n')[2:-1]:
            line = line.split(',')
            if line[2].split('/')[0].strip(' ') == 'NaN':
                continue
            SB2_IDs.append(IDline.split(',')[0][11:].strip(' '))
            SB2_dates.append(line[0].strip(' '))
            SB2_types.append(line[1].strip(' '))
            vguess1s.append(line[2].split('/')[0].strip(' '))
            vguess2s.append(line[2].split('/')[1].strip(' '))

if run:
    f  =open(f'{master_path}/Speciale/data/spectra_log_h_readable2.txt','w')
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
'''


targets  =open(f'{master_path}/Speciale/data/spectra_log_h_readable.txt').read().split('&')

print('The following new spectra have been observed:')
new_lines = ''
for target in targets[:-1]: #not including the last & since nothing after
    lines = target.split('\n')[:-1]
    nr_obs = int(lines[0].split(':')[1][1:3].strip(' '))
    
    target_ID = lines[0].split(',')[0][11:].strip(' ')
    print(target_ID)
    target_dates = []
    #print(target_dates)
    
    for line in lines[2:]:
        target_date = line.split(',')[0].strip(' ')
        target_dates.append(target_date)
    
    for i,ID in enumerate(IDs):
        if ID == target_ID:
            if dates[i] not in target_dates:
                print(dates[i])
                lines.append( f'  {dates[i]},         ,          ,           ,')
                nr_obs += 1

    lines[0] = lines[0].split(',')[0] + f', nr. obs: {nr_obs} ' +lines[0].split(',')[1][12:]

    for line in lines:
        new_lines += line +'\n'

    new_lines += '&'

    mt.add_value(nr_obs,'N_spec',target_ID)

    

f  =open(f'{master_path}/Speciale/data/spectra_log_h_readable.txt','w')
f.write(new_lines)
f.close()



    
    
            
        



















         
            
