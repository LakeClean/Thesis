import os
import glob
import sys

#KECK :
#BE CAREFUL RUNNING THIS CODE! WILL DELETE INFORMATION!

path = '/home/lakeclean/Documents/speciale/target_analysis/'

info_file = open('/home/lakeclean/Documents/speciale/KECK_order_file_log.txt').read().split('\n')

IDs = []
dates = []
for line in info_file[1:-1]:
    line = line.split(',')
    IDs.append(line[0].strip(' '))
    dates.append(line[2].strip(' ')[0:10])

for ID, date in zip (IDs, dates):
    print(path + ID + '/' + date + '/data/*')
    files = glob.glob(path + ID + '/' + date + '/data/*')
    for file in files:
        os.remove(file)

