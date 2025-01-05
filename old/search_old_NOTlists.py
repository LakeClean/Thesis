import make_table_of_target_info as mt
import glob
import pandas as pd
import numpy as np

path = '/home/lakeclean/Documents/speciale/lists_of_old_obs/'



list1 = path + 'FIES-FITSarchive.2018-2024.list.txt'
list2 = path + 'FIES-FITSarchive.2018.list.txt'

'''
FILENAME =[]
DATE_OBS=[]
EXPTIME= []
FIFMSKNM=[]
BIN =[]
OBJRA=[]
OBJDEC =[]
EQUIN=[]
AUSTATUS  =[]
TCSTGT    =[]
OBJECT    =[]
OBSERVER  =[]
PROPID  =[]
Reduced  =[]
SIMBAD_Object=[]
Distance = []
'''
tab = mt.get_table()

target_IDs = tab['ID'].data
target_RAs = tab['RA'].data
target_DECs = tab['DEC'].data

shortened = []
for ID in target_IDs:
    shortened.append(ID[4:])


lines1 = open(list1).read().split('\n')

header1 = lines1[41].split()

print(header1)
limit = 6/60

RA1s = []
DEC1s = []
pot1s = []
for i in lines1[43:-1]:
    i = i.split()

    RA1s.append(float(i[6]))
    DEC1s.append(float(i[7]))
    pot1s.append(''.join(i[10:]))


lines2 = open(list2).read().split('\n')

RA2s = []
DEC2s = []
pot2s = []
for i in lines2[41:-3]:
    i = i.split()
    
    RA2s.append(float(i[6]))
    DEC2s.append(float(i[7]))
    pot2s.append(''.join(i[10:]))

all_filenames = []
all_dates = []
all_reduced = []


k = 0
for t_ra,t_dec,ID,s_ID in zip(target_RAs,target_DECs,target_IDs,shortened):
    FILENAME =[]
    DATE_OBS=[]
    EXPTIME= []
    FIFMSKNM=[]
    BIN =[]
    OBJRA=[]
    OBJDEC =[]
    EQUIN=[]
    AUSTATUS  =[]
    TCSTGT    =[]
    OBJECT    =[]
    OBSERVER  =[]
    PROPID  =[]
    Reduced  =[]
    SIMBAD_Object=[]
    Distance = []
    
    output = ''
    for i, ra,dec,pot1 in zip(range(len(RA1s)),RA1s, DEC1s,pot1s):
        

        truth =False
        #if s_ID in pot1:
         #   truth = True
            
        
        if np.sqrt((ra - t_ra)**2+(dec - t_dec)**2)<limit:
            truth=True
            
            
        if truth:
                
            #output += lines1[43:-1][i]+'\n'

            line = lines1[41:-1][i].split()
            rest = ''.join(line[15:]).split('dist')

            value = input(f'{ID},{line[10]},{line[11]},{line[13]},{rest[0]}')
            if value=='y':
                k+=1

                
                FILENAME.append(line[0])
                DATE_OBS.append(line[1])
                EXPTIME.append(line[2])
                FIFMSKNM.append(line[3])
                BIN.append(' '.join((line[4],line[5])))
                OBJRA.append(line[6])
                OBJDEC.append(line[7])
                EQUIN.append(line[8])
                AUSTATUS.append(line[9])
                TCSTGT.append(line[10])
                OBJECT.append(line[11])
                OBSERVER.append(line[12])
                PROPID.append(line[13])
                Reduced.append(line[14])
                SIMBAD_Object.append(rest[0])
                Distance.append('dist'+rest[1])
    
    #if len(output)>0:
     #   f = open(path+f'{ID}_old_obs.txt','w')
      #  f.write(output)
       # f.close()




    #try:
     #   oldlines = open(path+f'{ID}_old_obs.txt').read()
    #except:
     #   oldlines = ''
        
    
    for i, ra,dec,pot2 in zip(range(len(RA2s)),RA2s, DEC2s,pot2s):
        truth = False
        #if s_ID in pot2:
        #    truth=True
        if (np.sqrt((ra - t_ra)**2+(dec - t_dec)**2)<limit):
            truth = True
        if truth:
                line = lines2[41:-1][i].split()
                
            
                
                k+=1
                #oldlines += lines2[41:-1][i]+'\n'


                
                FILENAME.append(line[0])
                print(line[0])
                DATE_OBS.append(line[1])
                EXPTIME.append(line[2])
                FIFMSKNM.append(line[3])
                BIN.append(' '.join((line[4],line[5])))
                OBJRA.append(line[6])
                OBJDEC.append(line[7])
                EQUIN.append(line[8])
                AUSTATUS.append(line[9])
                TCSTGT.append(line[10])
                OBJECT.append(line[11])
                OBSERVER.append(line[12])
                PROPID.append(line[13])
                Reduced.append(line[14])

                

                rest = ''.join(line[15:]).split('dist')
                SIMBAD_Object.append(rest[0])
                Distance.append('dist'+rest[1])
                print(ID,line[10],line[11],line[13],rest[0])
                
                

    #if len(oldlines)>0:        
     #   f = open(path+f'{ID}_old_obs.txt','w')
      #  f.write(oldlines)
       # f.close()

    all_filenames.append(FILENAME)
    all_dates.append(DATE_OBS)
    all_reduced.append(Reduced)
    out_dict = {}
    data = [FILENAME, DATE_OBS, EXPTIME, FIFMSKNM, BIN, OBJRA, OBJDEC, EQUIN,
        AUSTATUS, TCSTGT, OBJECT, OBSERVER, PROPID, Reduced, SIMBAD_Object, Distance]
    for head,dat in zip(header1,data):
        out_dict[head] = dat

    df = pd.DataFrame(out_dict)
    if len(out_dict['FILENAME'])>0:
        df.to_excel(path + f'old_obs_{ID}.xlsx',sheet_name='sheet1',index=False)
        

print(k)
print(all_filenames)

f = open('old_filenames.txt','w')
f.write('date, filename, reduced\n')
for files, dates,reduceds in zip(all_filenames,all_dates,all_reduced):
    for date,file,reduced in zip(dates,files,reduceds):
        f.write(f'{date}, {file}, {reduced}' + '\n')

f.close()

'''
FILENAME.append(i[0])
    DATE_OBS.append(i[1])
    EXPTIME.append(i[2])
    FIFMSKNM.append(i[3])
    BIN.append(' '.join((i[4],i[5])))
    OBJRA.append(i[6])
    OBJDEC.append(i[7])
    EQUIN.append(i[8])
    AUSTATUS.append(i[9])
    TCSTGT.append(i[10])
    OBJECT.append(i[11])
    OBSERVER.append(i[12])
    PROPID.append(i[13])
    Reduced.append(i[14])

    rest = ''.join(i[15:]).split('dist')
    SIMBAD_Object.append(rest)
    #Distance = []
'''
