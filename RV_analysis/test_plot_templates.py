import astropy.io.fits as pyfits
import pandas as pd
import matplotlib.pyplot as plt
from nsstools import NssSource
import make_table_of_target_info as mt
import numpy as np
import glob
import shazam
from astropy.time import Time
from PyAstronomy import pyasl #For converting from vacuum to air
import read_goettingen as gg

def wl_corr(wl,vbary=0):
    c = 299792 #speed of light km/s
    return (1+vbary/c)*wl

def vac2air(wl_vac):
    s = 10**4 / wl_vac
    n = 1 +  0.0000834254 + 0.02406147 / (130 - s**2)
    n += 0.00015998 / (38.9 - s**2)
    return wl_vac / n
    

#Plotting the templates to compare them

#Old template from arcturus:
fig,ax =plt.subplots()
template_dir = '~/Speciale/data/templates/ardata.fits'
template_data = pyfits.getdata(f'{template_dir}')
tfl_RG = template_data['arcturus']
tfl_MS = template_data['solarflux']
twl = template_data['wavelength']


#from goettingen

#path = '/home/lakeclean/Downloads/spvis.dat.gz'

#df = pd.read_csv(path,delim_whitespace=True)

#print(df.keys())

#df = df.to_numpy()


#also from goettingen:
wn, sun, tel, wave = gg.get_goettingen()
plt.plot(wl_corr(wave,0),sun,color='r')

plt.plot(wl_corr(wave,0.15),sun,color='b')


#plt.plot(pyasl.vactoair2(10**(8)/df[:,0]),df[:,1],color='r',label='(Reiners, 2016)')
#plt.plot(vac2air(10**(8)/df[:,0]),df[:,1],color='pink',
#         label='(Reiners, 2016)')

#ax.plot(wl_corr(twl,0),tfl_MS,color='r')
#ax.plot(wl_corr(twl,0.28),tfl_MS,color='b')
plt.xlim(6767.6,6768)
plt.ylim(0.3,1.1)
#plt.show()
plt.close()

# Looking at the KECK spectra to see if offset is similar to RV found.

rv_path = '~/Speciale/data/rv_data/KECK_KIC10454113.txt'


df = pd.read_csv(rv_path)
dates = df['date'].to_numpy()
rvs = df['rv1'].to_numpy()
vbarys = df['vbary'].to_numpy()


path_folders = '~/Speciale/data/initial_data/KECK/KIC10454113'
folders = glob.glob(path_folders + '/*')

def sorter(folder):
    return folder[len(path_folders)+1:]

folders.sort(key=sorter)

for folder in folders:
    files = glob.glob(folder + '/*')
    
    
    for i,date in enumerate(dates):
        if date[0:10] == folder[len(path_folders)+1:]:
            rv = rvs[i]
            vbary = vbarys[i]
            print(date[0:10],folder[len(path_folders)+1:],rv,vbary,rv-vbary)
    
    for file in files:
        data = pyfits.getdata(file)
        w = data['wave']
        #w = pyasl.vactoair2(w)
        w = wl_corr(w,-rv)
        w = wl_corr(w,vbary)

        #if (w[0] < 6500) & (w[0]>6000):
        f = data['Flux']
        nw, nf = shazam.normalize(w,f,gauss=False)
        plt.plot(nw,nf,color='red')
#plt.plot(wave,sun)
#plt.xlim(4850,4865)
plt.ylim(-0.5,2)
#plt.show()
 
   

#new templates from Phoenix:
#plt.close()

phoenix_templates = '~/Speciale/data/templates/phoenix/'
wl = pyfits.getdata(phoenix_templates + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
fl = pyfits.getdata(phoenix_templates + 'lte04700-3.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
fl = pyfits.getdata(phoenix_templates + 'lte06100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')



idx = np.where( (wl>3000) & (wl <16900))[0]
#wl = pyasl.vactoair2(wl[idx])
wl = wl[idx]
fl = fl[idx]/np.amax(fl[idx])

#fig,ax = plt.subplots()
for i in range(100):
    i = i*100
    tmp_idx = np.where( ((wl[0] + i )<wl) & ((wl[0] + i +100 )>wl))[0]
    nwl, nfl = shazam.normalize(wl[tmp_idx],fl[tmp_idx],gauss=True)
    plt.plot(nwl,nfl,color='b')
    #ax.plot(wl[tmp_idx],fl[tmp_idx],color='r')
#ax.plot(wave,sun,color='b')
plt.show()


#Comparing the values found by me and those from Gaia:

'''
nss = pd.read_csv("/home/lakeclean/Downloads/Jonatan.csv")
for i in range(4):
    source = NssSource(nss, indice=i)
    print(source.covmat())
    print(source.campbell())

tab = mt.get_table()
for i in tab['G_i'].data:
    print(np.degrees(i))
'''


#Showing that TNG red and blue is weird:
'''
blue_path = '~/Special/data/initial_data/TNG/2016-07-19/KIC104.blue.norm.fits'
red_path = '~/Special/data/initial_data/TNG/2016-07-19/KIC104.red.norm.fits'
regular_path = '~/Special/data/target_analysis/KIC10454113/2016-07-19T22:19:33.321/data/order_*_normalized.txt'


red_data = pyfits.getdata(red_path)
red_header = pyfits.getheader(red_path)
red_lam = red_header['CDELT1']*np.arange(red_header['NAXIS1'])+red_header['CRVAL1']


blue_data = pyfits.getdata(blue_path)
blue_header = pyfits.getheader(blue_path)
blue_lam = blue_header['CDELT1']*np.arange(blue_header['NAXIS1'])+blue_header['CRVAL1']

#plt.plot(red_lam+blue_lam[-1]-red_header['CRVAL1'],red_data,color='red')
plt.plot(blue_lam,blue_data,color='blue')


regular_files = glob.glob(regular_path)
for file in regular_files:
    df = pd.read_csv(file)
    regular_lam = df['wavelength [Å]']
    regular_data = df['flux (norm)']
    plt.plot(regular_lam,regular_data,color='orange')
    
plt.show()
'''

#Plotting KIC1045... from KECK to see if there is relative movement.
'''
rv_path = '~/Special/data/target_analysis/KIC10454113'

file_path = '~/Special/data/KECK_ordered_file_log.txt'

df = pd.read_csv(file_path)
dates = df['date'].to_numpy()
v_barys = df['v_bary'].to_numpy()
jds = df['epoch_jd'].to_numpy()


vrad = []
for date,v_bary in zip(dates,v_barys):
    simple_date = date.split('T')[0]

    df = pd.read_csv(f'{rv_path}/{simple_date}/data/bf_fit_params.txt')
    vrad.append(np.median(df['epoch_vrad1'].to_numpy()))
    
plt.scatter(jds,vrad)
plt.show()
'''

#Plotting the extreme case of KIC1045...



'''
vbary = -7406.861379294089/1000 -12

order_files = glob.glob('~/Special/data/target_analysis/KIC10454113/2011-09-05T00:58:25.2/data/order_*_normalized.txt')
for file in order_files:
    df = pd.read_csv(file,skiprows=1).to_numpy()
    plt.plot(wl_corr(df[:,0],vbary),df[:,1])
plt.show()
'''

#Plotting every stars spectrum

'''

rv_path = '~/Special/data/rv_data/*'
spectra_path = '~/Special/data/target_analysis/'
stars = glob.glob(rv_path)


tab = mt.get_table()
all_IDs = tab['ID'].data
print(all_IDs)

for star in stars:
    name = star[len(rv_path)-1:]
    
    df = pd.read_csv(star)
    date = df['date'].to_numpy()[0]
    rv1 = df['rv1'].to_numpy()[0]
    vbary = df['vbary'].to_numpy()[0]
    k = 0
    for ID in all_IDs:
        if ID in star:
            #print(ID,name)
            path = spectra_path + ID + f'/{date}/data/order_*_normalized.txt'
            files = glob.glob(path)
            for file in files:
                df = pd.read_csv(file).to_numpy()
                plt.plot(wl_corr(df[:,0],rv1-vbary),df[:,1] + k, color='black',label=name)
        k+=0.3

plt.xlim(6077,6090)
plt.ylim(-1,6)
plt.show()
           
'''

#riverplot for KIC1045

logs = ['KECK_ordered_file_log.txt', 'NOT_old_HIRES_order_file_log.txt',
        'NOT_old_LOWRES_order_file_log.txt', 'NOT_order_file_log.txt',
        'TNG_merged_file_log.txt']
'''
type_directory = []
type_vbary = []
type_jd = []


for log in logs:
    df = pd.read_csv('~/Special/data/'+log)
    IDs = df['ID'].to_numpy()
    directorys = df['directory'].to_numpy()
    vbarys = df['v_bary'].to_numpy()
    jds = df['epoch_jd'].to_numpy()
    index = np.where(IDs == 'KIC10454113')[0]

    type_directory.append(directorys[index])
    type_vbary.append(vbarys[index])
    type_jd.append(jds[index])
    
    if log[0] == 'N':
        for directory, vbary, jd in zip(directorys[index],vbarys[index],jds[index]):
                data, no_orders, bjd, vhelio, star, date, exp = shazam.FIES_caliber(directory)
                for i in np.arange(1,no_orders,1):
                    #Pick out correct wl range
                    wl, fl = shazam.getFIES(data,order=i)
                   plt.plot(wl_corr(wl),fl+(jd-2457000))
'''


'''
k = 0.001
fig,ax = plt.subplots(figsize=(4,12))


log = logs[-1]
df = pd.read_csv('~/Special/data/'+log)
IDs = df['ID'].to_numpy()
directorys = df['directory'].to_numpy()
vbarys = df['v_bary'].to_numpy()
jds = df['epoch_jd'].to_numpy()
index = np.where(IDs == 'KIC10454113')[0]
index = [index[0]]

count=0
for directory, vbary, jd in zip(directorys[index],vbarys[index],jds[index]):
        data = pyfits.getdata(directory)

        header = pyfits.getheader(directory)
        lam = header['CDELT1']*np.arange(header['NAXIS1'])+header['CRVAL1']
        data = data[(lam > 6540) & (lam<6580)]

        lam = lam[ (lam > 6540) & (lam<6580)]
        nwl ,nfl = shazam.normalize(lam,data,poly=1,gauss=False)
        if count==0:
            ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='green',label='TNG')
        else:
            ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='green')
        count+=1
            

        


            

log = logs[3]
df = pd.read_csv('~/Special/data/'+log)
IDs = df['ID'].to_numpy()
directorys = df['directory'].to_numpy()
vbarys = df['v_bary'].to_numpy()
jds = df['epoch_jd'].to_numpy()
index = np.where(IDs == 'KIC10454113')[0]
index = [index[0],index[-2]]

count=0
for directory, vbary, jd in zip(directorys[index],vbarys[index],jds[index]):
        data, no_orders, bjd, vhelio, star, date, exp = shazam.FIES_caliber(directory)
        for i in np.arange(1,no_orders,1):
            #Pick out correct wl range
            wl, fl = shazam.getFIES(data,order=i)
            nwl ,nfl = shazam.normalize(wl,fl,poly=1,gauss=False)
            if count==0:
                ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='red',label='new NOT')
            else:
                ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='red')
            count+=1


log = logs[1]
df = pd.read_csv('~/Special/data/'+log)
IDs = df['ID'].to_numpy()
directorys = df['directory'].to_numpy()
vbarys = df['v_bary'].to_numpy()
jds = df['epoch_jd'].to_numpy()
index = np.where(IDs == 'KIC10454113')[0]
#index = [index[0],index[-2]]
count=0
for directory, vbary, jd in zip(directorys[index],vbarys[index],jds[index]):
        data, no_orders, bjd, vhelio, star, date, exp = shazam.FIES_caliber(directory)
        for i in np.arange(1,no_orders,1):
            #Pick out correct wl range
            wl, fl = shazam.getFIES(data,order=i)
            nwl ,nfl = shazam.normalize(wl,fl,poly=1,gauss=False)

            if count==0:
                ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='blue')
            else:
                ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='blue')
            count+=1
                


log = logs[2]
df = pd.read_csv('~/Special/data/'+log)
IDs = df['ID'].to_numpy()
directorys = df['directory'].to_numpy()
vbarys = df['v_bary'].to_numpy()
jds = df['epoch_jd'].to_numpy()
index = np.where(IDs == 'KIC10454113')[0]
#index = [index[0],index[-2]]
count=0
for directory, vbary, jd in zip(directorys[index],vbarys[index],jds[index]):
        data, no_orders, bjd, vhelio, star, date, exp = shazam.FIES_caliber(directory)
        for i in np.arange(1,no_orders,1):
            #Pick out correct wl range
            wl, fl = shazam.getFIES(data,order=i)
            nwl ,nfl = shazam.normalize(wl,fl,poly=1,gauss=False)
            if count==0:
                ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='blue',label='old NOT')
            else:
                ax.plot(wl_corr(nwl,float(vbary)/1000),
                    nfl+(jd-2457000)*k,color='blue')
            count+=1
                

print(wl_corr(6562.8,-20))
ax.plot([wl_corr(6562.8,-20),wl_corr(6562.8,-20)],[-2, 5],
        ls='--',alpha=0.4,color='black',label=f'{wl_corr(6562.8,-20)}Å')
ax.set_ylim(-2, 5)
ax.set_xlim(6545,6580)
ax.set_xlabel('Wavelength [Å]')
ax.set_ylabel(f'fl+(jd-2457000)*{k}')
ax.set_title('Spectrum corrected for V_bary')
ax.legend()
plt.show()

save_path='~/Special/data/random_plots/'
save_path += 'KIC10454113_spectrum_shows_rv_motion.pdf'
fig.savefig(save_path,dpi='figure', format='pdf')
'''


#log for vbarys from date
log_path = '~/Special/data/'
logs = ['NOT_old_HIRES_order_file_log.txt',
        'NOT_old_LOWRES_order_file_log.txt', 'NOT_order_file_log.txt',
        'TNG_merged_file_log.txt']

vbarys = {}
telescopes = {}
for log in logs:
    df = pd.read_csv(log_path + log)
    directory = df['directory'].to_numpy()
    date = df['date'].to_numpy()
    vbary = df['v_bary'].to_numpy()
    telescope = df['TELESCOP'].to_numpy()
    for i in range(len(vbary)):
        #date = directory[i].split('\\')[-1]
        vbarys[f'{date[i]}'] = float(vbary[i])/1000
        telescopes[f'{date[i]}'] = telescope[i]
    


#Plotting the BF of KIC10454113 to show the SB2 nature
'''
path = '~/Special/data/target_analysis/KIC10454113/*'
path = '~/Special/data/target_analysis/KIC4914923/*'
folders = glob.glob(path)
def sorter(folder):
    date = folder[len(path)-1:]
    jd = Time(date).jd
    return jd

folders.sort(key=sorter)


count = 0
fig, ax = plt.subplots()
selected = [1,2,3]
for folder in folders:
    
    date = folder[len(path)-1:]
    print(date)
    

    jd = Time(date).jd
    files = glob.glob(folder + '/data/order_*_broadening_function.txt')
    files.sort()

    if len(str(date)) < len('2016-07-16')+1:
        continue

    if telescopes[date]=='TNG':
        color='green'
        start = 2
        end = 12
    if telescopes[date]=='NOT':
        color='purple'
        start = 10
        end = 30
        
    df = pd.read_csv(files[start],skiprows=1).to_numpy()
    summed_bf = df[:,2] #Smoothed bf
    rvs = df[:,0]+vbarys[date]
    print(vbarys[date])
    
    for i in np.arange(start+1,end,1):
        df = pd.read_csv(files[i],skiprows=1).to_numpy()
        summed_bf += df[:,2]
        
    summed_bf /=(end-start)
    ax.plot(rvs,summed_bf+count, color=color)
    count+=0.08


ax.set_ylim(-5,5)
ax.set_xlabel('RV [km/s]')
ax.set_ylabel('BF + offset')
plt.show()        
save_path='~/Special/data/random_plots/'
save_path += 'KIC10454113_old_NOT_SB2_BFs.pdf'
fig.savefig(save_path,dpi='figure', format='pdf')
'''



#Fitting extreme point for KIC1045
'''
path = '~/Special/data/target_analysis/KIC10454113/'
date = '2011-09-05T00:58:25.2'
#date = '2016-04-27T03:51:43.5'
path += date
jd = Time(date).jd
files = glob.glob(path + '/data/order_*_broadening_function.txt')
files.sort()
fig,ax =plt.subplots()
df = pd.read_csv(files[0],skiprows=1).to_numpy()
summed_bf = df[:,2] #Smoothed bf
rvs = df[:,0]#+vbarys[date]
count=0
for i in np.arange(1,len(files),1):
    df = pd.read_csv(files[i],skiprows=1).to_numpy()
    summed_bf += df[:,2]
    #plt.plot(df[:,0], df[:,2]+ count,color='black')
    count += 0.09

#Fitting:
fit, model, bfgs = shazam.rotbf2_fit(rvs,summed_bf/len(files),
                                     30,60000,1,5,5,-33,
                                     -12,0.05,0.05,False, False)
vrad1,vrad2 = fit.params['vrad1'].value, fit.params['vrad2'].value
ampl1, ampl2 = fit.params['ampl1'].value,fit.params['ampl2'].value
print(vrad1+vbarys[date],vrad2+vbarys[date])
print(jd - 2457000)
ax.plot(rvs,model,color='red')

ax.set_title('Summed BF KIC10454113')
ax.set_xlabel('RV [km/s]')
ax.set_ylabel('Summed BF')
ax.plot(rvs, summed_bf/(len(files)),color='black')   
plt.show()

#save_path='~/Special/data/random_plots/KIC4914923_sum_BF_HIGH_EXPOSURE.pdf'
#fig.savefig(save_path,dpi='figure', format='pdf')
'''


#Checking the high exposure obs of KIC49 (BF)
'''
path = '~/Special/data/target_analysis/KIC4914923/'
date = '2016-04-27T03:53:16.9'
#date = '2016-04-27T03:51:43.5'
path += date
jd = Time(date).jd
files = glob.glob(path + '/data/order_*_broadening_function.txt')
files.sort()
fig,ax =plt.subplots()
df = pd.read_csv(files[6],skiprows=1).to_numpy()
summed_bf = df[:,2] #Smoothed bf
rvs = df[:,0]+vbarys[date]
count=0
for i in np.arange(7,50,1):
    df = pd.read_csv(files[i],skiprows=1).to_numpy()
    summed_bf += df[:,2]
    #plt.plot(df[:,0], df[:,2]+ count,color='black')
    #plt.plot(df[:,0], df[:,2]+ count,color='black')
    count += 0.09

ax.set_title('Summed BF KIC4914923, Texp=2400s')
ax.set_xlabel('RV [km/s]')
ax.set_ylabel('Summed BF')
ax.plot(df[:,0]+vbarys[date], summed_bf/(50-6),color='black')   
plt.show()

save_path='~/Special/data/random_plots/KIC4914923_sum_BF_HIGH_EXPOSURE.pdf'
fig.savefig(save_path,dpi='figure', format='pdf')
'''


#Checking the high exposure obs of KIC49 (spectre)
'''
path = '~/Special/data/target_analysis/KIC4914923/'
date = '2016-04-27T03:53:16.9'
#date = '2016-04-27T03:51:43.5'
path += date
jd = Time(date).jd
files = glob.glob(path + '/data/order_*_normalized.txt')
files.sort()
fig, ax = plt.subplots()
print(wl_corr(6000,vbarys[date]),vbarys[date])
count=0
for i in np.arange(7,70,1):
    df = pd.read_csv(files[i],skiprows=1).to_numpy()

    ax.plot(wl_corr(df[:,0],vbarys[date]), df[:,1],color='black')



#ax.set_title('Summed BF KIC4914923, Texp=2400s')
ax.set_xlabel('wavelength [Å]')
ax.set_ylabel('norm flux')
#ax.plot(df[:,0]+vbarys[date], summed_bf/(50-6),color='black')



save_path='~/Special/data/speciale/random_plots/'
save_path += 'KIC4914923_HIGH_EXPOSURE_spectrum.pdf'
fig.savefig(save_path,dpi='figure', format='pdf')
plt.show()
'''


#Plotting the odd spectrum of KIC4914923
'''
path = '~/Special/data/target_analysis/KIC4914923/'
date = '2013-08-06T23:33:46.4'

path += date
jd = Time(date).jd
files = glob.glob(path + '/data/order_*_normalized.txt')
files.sort()
fig, ax = plt.subplots()

count=0
for i in np.arange(7,70,1):
    df = pd.read_csv(files[i],skiprows=1).to_numpy()

    ax.plot(wl_corr(df[:,0],vbarys[date]), df[:,1],color='black')



#ax.set_title('Summed BF KIC4914923, Texp=2400s')
ax.set_xlabel('wavelength [Å]')
ax.set_ylabel('norm flux')
#ax.plot(df[:,0]+vbarys[date], summed_bf/(50-6),color='black')



#save_path='~/Special/data/random_plots/'
#save_path += 'KIC4914923_HIGH_EXPOSURE_spectrum.pdf'
#fig.savefig(save_path,dpi='figure', format='pdf')
plt.show()
'''














