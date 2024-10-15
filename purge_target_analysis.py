import os
import glob
import sys
path = '/home/lakeclean/Documents/speciale/target_analysis/'

info_file = open('file_log.txt').read().split('\n')

IDs = []
dates = []
for line in info_file[1:-1]:
    line = line.split(',')
    if line[0].strip() not in IDs:
        IDs.append(line[0].strip())
    dates.append(line[3].strip())

name = sys.argv[1:]

if name[0] == 'all':
    for ID in IDs:
        dates = glob.glob(path+f'{ID}/*')
        for date in dates:
            data_files = glob.glob(date + '/data/*')
            plots = glob.glob(date + '/plots/*')
            
            for file in data_files:
                os.remove(file)

            for plot in plots:
                os.remove(plot)

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
        
            

