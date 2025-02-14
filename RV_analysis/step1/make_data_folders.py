import os
import glob
import sys
master_path = '/usr/users/au662080'

#Making folders for data to be filled by other script:

#NOT data:

path = f'{master_path}/Speciale/data/target_analysis/'

info_file = open(f'{master_path}/Speciale/data/NOT_order_file_log.txt').read().split('\n')

IDs = []
dates = []
for line in info_file[1:-1]:
    line = line.split(',')
    IDs.append(line[0].strip(' '))
    dates.append(line[2].strip(' '))
    


added = []
old = []

for ID, date in zip (IDs, dates):
    try:
        os.mkdir(path + ID + '/' + date)
        os.mkdir(path + ID + '/' + date + '/data')
        os.mkdir(path + ID + '/' + date + '/plots')
        added.append([ID,date])
    except:
        old.append([ID,date])
        pass
                
                #for file in data_files:
                 #   os.remove(file)

                #for plot in plots:
                 #   os.remove(plot)

print('############### NOT files: ###################')
print('These were added')
for i in added:
    print(i[0],i[1])

print('These already existed')
for i in old:
    print(i[0],i[1])


#TNG (HARPS):
path = f'{master_path}/Speciale/data/target_analysis/'

info_file = open(f'{master_path}/Speciale/data/TNG_merged_file_log.txt').read().split('\n')

IDs = []
dates = []
for line in info_file[1:-1]:
    line = line.split(',')
    IDs.append(line[0].strip(' '))
    dates.append(line[2].strip(' '))
        
#if line[0].strip() not in IDs:
#name = sys.argv[1:]

#if name[0] == 'all':
'''
done_IDs = []
for ID in IDs:
    if ID not in done_IDs:
        os.mkdir(path + ID)
        done_IDs.append(ID)
'''


added = []
old = []

for ID, date in zip (IDs, dates):
    try:
        os.mkdir(path + ID + '/' + date)
        os.mkdir(path + ID + '/' + date + '/data')
        os.mkdir(path + ID + '/' + date + '/plots')
        added.append([ID,date])
    except:
        old.append([ID,date])
        pass
                
                #for file in data_files:
                 #   os.remove(file)

                #for plot in plots:
                 #   os.remove(plot)
print('############### TNG (HARPS) files: ###################')
print('These were added')
for i in added:
    print(i[0],i[1])

print('These already existed')
for i in old:
    print(i[0],i[1])

'''

if name[0] != 'all' and len(name) == 1:
    if name in IDs:
        try:
            for ID in IDs:
                for date in dates:
                    os.remove(path + '/' +  ID + '/' + date + '/data/*')
                    os.remove(path + '/' +  ID + '/' + date + '/plots/*')
        except:
            print('mistake')

    else:
        print('The ID given does not exist')

'''
#KECK :
path = f'{master_path}/Speciale/data/target_analysis/'

info_file = open(f'{master_path}/Speciale/data/KECK_order_file_log.txt').read().split('\n')

IDs = []
dates = []
for line in info_file[1:-1]:
    line = line.split(',')
    IDs.append(line[0].strip(' '))
    dates.append(line[2].strip(' ')[0:10])


added = []
old = []

for ID, date in zip (IDs, dates):
    try:
        os.mkdir(path + ID + '/' + date)
        os.mkdir(path + ID + '/' + date + '/data')
        os.mkdir(path + ID + '/' + date + '/plots')
        added.append([ID,date])
    except:
        old.append([ID,date])
        pass
                
print('############### KECK (HIRES) files: ###################')
print('These were added')
for i in added:
    print(i[0],i[1])

print('These already existed')
for i in old:
    print(i[0],i[1]) 
         


#CFHT (ESpaDOns):
path = f'{master_path}/Speciale/data/target_analysis/'

info_file = open(f'{master_path}/Speciale/data/ESpaDOns_merged_file_log.txt').read().split('\n')

IDs = []
dates = []
for line in info_file[1:-1]:
    line = line.split(',')
    IDs.append(line[0].strip(' '))
    dates.append(line[2].strip(' '))
        


added = []
old = []

for ID, date in zip (IDs, dates):
    try:
        os.mkdir(path + ID + '/' + date)
        os.mkdir(path + ID + '/' + date + '/data')
        os.mkdir(path + ID + '/' + date + '/plots')
        added.append([ID,date])
    except:
        old.append([ID,date])
        pass
                
                #for file in data_files:
                 #   os.remove(file)

                #for plot in plots:
                 #   os.remove(plot)
print('############### CFHT (ESpaDOns) files: ###################')
print('These were added')
for i in added:
    print(i[0],i[1])

print('These already existed')
for i in old:
    print(i[0],i[1])










