import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ps import powerspectrum
from bottleneck import nanmedian, nanmean, nanmax, nanmin
from scipy.interpolate import InterpolatedUnivariateSpline as INT
from scipy import optimize as OP
from matplotlib.colors import LogNorm
from matplotlib import cm
import glob
folder_path = '~/Special/data/Seismology/data/'

###### 1) #######
#Importing lightcurve data:
'''
df = pd.read_csv(folder_path + 'kplr004260884_kasoc-ts_llc_v2.dat',
                 skiprows=13,
                 delimiter=r"\s+").to_numpy()

time = df[:,0]
flux = df[:,1]
e_flux = df[:,2]
'''

#Plotting timeseries
'''
fig, ax = plt.subplots()
ax.plot(time,flux)
ax.set_xlabel('Truncated julean time')
ax.set_ylabel('relative flux ppm')
plt.show()
'''

####### 2) #######
#We make a rough estimate of frequency spacing as 1/(max_time -min_time)
'''
day_in_sec = 86400 # seconds
df = 1/( day_in_sec *(max(time) - min(time))) #Hz
f_ny = 1/(2*day_in_sec*np.median(np.diff(time))) # Hz 86400s = 1day
print('estimate of frequency spacing:', df)
print('estimate of nyquist frequency:', f_ny)
'''


####### #3) #######
#Finding power spectrum:
'''
psd = powerspectrum(time,flux,flux_err=e_flux,weighted=True)
nu, power = psd.powerspectrum(scale='power')
'''
#Plotting powerspectrum:
'''
fig, ax = plt.subplots()
ax.plot(nu,power)
ax.set_ylabel('powerspectrum')
ax.set_xlabel('frequency [muHz]')
ax.set_yscale('log')
plt.show()
'''

####### 4-5) #######
#We generate a sine wave:
def sine_wave(time,A=1, f=1, phi=np.pi/2):
    return A*np.sin(2*np.pi*f*time + phi)
'''
#timeseries of duration 20pi
time = np.arange(-10*np.pi,10*np.pi,0.01)
flux = sine_wave(time,np.pi/10,1/(2*np.pi))
psd = powerspectrum(time,flux)
nu,power = psd.powerspectrum(scale='powerdensity',oversampling=100)

#timeseries of duration 2pi
time2 = np.arange(-1*np.pi,1*np.pi,0.01)
flux2 = sine_wave(time2,np.pi/10,1/(2*np.pi))
psd2 = powerspectrum(time2,flux2)
nu2,power2 = psd2.powerspectrum(scale='powerdensity',oversampling=100)
'''
#plotting
'''
fig,ax = plt.subplots(1,2)
ax[0].plot(time,flux)
ax[0].set_xlabel('Time(days)')
ax[0].set_ylabel('Flux')

ax[1].plot(nu,power)

ax[0].plot(time2,flux2)
ax[1].plot(nu2,power2)
plt.show()
'''

'''
What is clear from the change in duration of the time series is that
with a longer timeseries the peaks of the powerdensity become much sharper
and more numerous. The thought is maybe therefore that the timeseries
should be as long as possible to increase the sharpness of the peaks.
'''


####### 6) #######
#simple filtering:


def filtering(signal,win,filt='Epanechnikov'):
    '''
    signal: signal to be filtered
    win: window width in number of points:
    filt: the type of filter (window) to be used
    '''

    def Epanechnikov(win):
        u = np.array(range(win)) - np.array(range(win)).max()/2
        u /= np.array(range(win)).max()/2
        return 3/4 * (1 - u**2)

    def Tricube(win):
        u = np.array(range(win)) - np.array(range(win)).max()/2
        u /= np.array(range(win)).max()/2
        return 70/81 * (1 - abs(u)**3)**3
    
    def Uniform(win):
        u = np.array(range(win)) - np.array(range(win)).max()/2
        u /= np.array(range(win)).max()/2
        return 1/2
        
        
    filters = ['Epanechnikov', 'Tricube', 'Uniform']
    options = [Epanechnikov,Tricube,Uniform]
    for i in range(len(filters)):
        if filters[i]==filt:
            window = options[i]
            break
            
    w = window(win)



    sig = np.r_[signal[int(win)-1:0:-1], signal, signal[-1:-int(win):-1] ]
    y = np.convolve(w/w.sum(),sig,mode='valid')

    y = y[int(np.floor(win/2)):len(y)-int(np.floor(win/2))]

    return y

def sigma_clip(time,flux,sigma):
    time = np.asarray(time)
    flux = np.asarray(flux)
    mask = np.ones(len(flux),dtype=bool)
    while True:
        mean_flux = np.mean(flux[mask])
        std_flux = np.std(flux[mask])
        new_mask = np.abs(flux-mean_flux)<sigma*std_flux
        if np.array_equal(mask,new_mask):
            break
        mask = new_mask
    return time[mask], flux[mask]

#Importing lightcurve data:
df = pd.read_csv(folder_path + 'kplr004260884_kasoc-ts_llc_v2.dat',
                 skiprows=13,
                 delimiter=r"\s+").to_numpy()
time = df[:,0]
flux = df[:,1]
e_flux = df[:,2]



#plotting
'''
fig,ax = plt.subplots()
ax.plot(time,flux)
ax.plot(time[1:],filtering(flux,100))
plt.show()
'''

#Comparing psd with and without filter
'''
filt_flux = filtering(flux,10)
filt_time = time[:len(filt_flux)]


psd = powerspectrum(time,flux)
nu,power = psd.powerspectrum(scale='powerdensity',oversampling=1)

filt_psd = powerspectrum(filt_time,filt_flux)
filt_nu,filt_power = filt_psd.powerspectrum(scale='powerdensity',
                                            oversampling=1)

fig,ax =plt.subplots()
ax.plot(nu,power)
ax.plot(filt_nu,filt_power)
plt.show()
'''





#########################################################################
# Code examples from Mikkel (parameters):
#########################################################################

def logmed_filter(frequency,power,filter_width=0.01):
    
    count = np.zeros(len(frequency),dtype=int)
    bkg = np.zeros_like(frequency)
    x0 = np.log10(frequency[0])
    logf = np.log10(frequency)
    corr_factor = (8.0 / 9.0)**3
    while x0 < np.log10(frequency[-1]):
        m = (np.abs(logf - x0) < filter_width)
        if len(bkg[m]) > 0:
            bkg[m] += nanmedian(power[m]) / corr_factor
            count[m] += 1
            x0 += 0.5 * filter_width
        else:
            print('Choose a larger filter with')
            break
    
    smooth_pg = bkg / count
    return smooth_pg

'''
fig, ax = plt.subplots()
#pds = powerspectrum(time,flux)
#ax.plot(

#N = len(time)
#time, flux = time[0:int(N/2)], flux[0:int(N/2)]
psd = powerspectrum(time,flux)
f, p = psd.powerspectrum(scale='powerdensity')

s_p = logmed_filter(f,p,0.1)


ax.plot(f,p)
ax.plot(f,s_p)
plt.show()
'''

def autocorr_fft(x):
    xp = x-np.nanmean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:int(x.size/2)]/np.sum(xp**2)

def autocorr(x):
    X = x- np.nanmean(x)
    result = np.correlate(X,X,mode='same')
    return result[int(result.size/2):]


def find_maximum(guess,X,Y,minx,maxx):
    fx = INT(X,Y)
    def opt(x): return -fx(x)
    opt = OP.minimize(opt, guess,method='Nelder-Mead', bounds=[(minx,maxx)])
    return opt.x

def dnu_guess(numax):
    alpha_u, beta_u = 0.267, 0.769
    alpha_o, beta_o = 0.22, 0.797

    Dnu_guess = alpha_u * ( numax<300)*numax**beta_u
    Dnu_guess += alpha_o * ( numax>=300)*numax**beta_o
    return Dnu_guess
    


def Echellediagram(Power, Fre,lsep,Start,Stop,no_repeat=1):

    sta = np.argmin(np.abs(Fre-Start))
    sto = np.argmin(np.abs(Fre -Stop))
    step = np.median(np.diff(Fre))
    Lsep_len = int(round(lsep/step))
    Lsep_len2 = int(round(lsep/step))*no_repeat
    N = int(np.ceil((len(Fre[sta:sto])/Lsep_len)))

    ValsP = np.zeros([N,Lsep_len2])
    FreVec_y = np.zeros(N)
    for i in np.arange(N):
        if i==0:
            start = sta
            stop = sta + Lsep_len2
            FreVec_y[i] = lsep/2 + Start
            FreVec_x = np.mod(Fre[start:stop], lsep*no_repeat)
            idx_sort = np.argsort(FreVec_x)
            FreVec_y[i] = FreVec_y[i-1] + lsep
        else:
            start = stop - Lsep_len*(no_repeat-1)
            stop = int(start + Lsep_len2)
            FreVec_y[i] = FreVec_y[i-1] + lsep

        ValsP[i,:] = Power[start:stop][idx_sort]

    EchelleVals = ValsP[::-1]
    return FreVec_x, FreVec_y, EchelleVals
    
#########################################################################
#Testing:
#########################################################################

datasets = glob.glob(folder_path + '*.pow')
numaxs = [3000,2250,1200,1800]
guess_dnus = [132,100,100,100,100]
for dataset,numax,dnu_guess in zip(datasets,numaxs,guess_dnus):
    lines = open(dataset).read().split('\n')
    ID = lines[2]
    print('Analysing ',ID, '|  numax guess:', numax, '|  dnu guess:', dnu_guess)
    df = pd.read_csv(dataset,skiprows=13,delimiter=r"\s+").to_numpy()

    f = df[:,0] #muHz
    p = df[:,1] #ppm**2 / muHz

    #### plotting full powerspectrum: ####
    '''
    fig, ax = plt.subplots()
    ax.plot(f,p)
    ax.set_xlabel(r'frequency [$\mu$Hz]')
    ax.set_ylabel(r'power [ppm^2 / $\mu$Hz]')
    ax.set_title(ID)
    plt.show()
    '''

    #### Finding ACF: ####

    #The powerspectrum appears already to be background corrected...
    #So we take section around what appears to be numax ~1200 muHz and find ACF
    #numax = 1200 #muHz
    HWHM = numax/4
    idx1 = (f<numax + 1.5*HWHM) & (f> numax -1.5*HWHM)
    idx2 = (f<numax + 3*HWHM) & (f> numax -3*HWHM)


    #weight?
    env_scale=1
    sigma_env = env_scale * numax/(4*np.sqrt(2*np.log(2)))
    weight = 1 / (sigma_env * np.sqrt(2*np.pi)) * np.exp(-(f - numax)**2 / (2*sigma_env**2) )
    pds_w = p*weight

    acf1 = autocorr_fft(p[idx1])
    acf2 = autocorr_fft(p[idx2])


    acf1_w = autocorr_fft(pds_w[idx1])
    acf2_w = autocorr_fft(pds_w[idx2])

    df = f[10] - f[9]
    lagvec1 = np.arange(len(acf1))*df
    lagvec2 = np.arange(len(acf2))*df

    #Finding maximums in ACF:
    dnu_peak1 = find_maximum(dnu_guess,lagvec1,acf1/acf1[1],dnu_guess*0.9,dnu_guess*1.1)[0]
    dnu_peak2 = find_maximum(dnu_guess,lagvec2,acf2/acf2[1],dnu_guess*0.9,dnu_guess*1.1)[0]
    

    #plotting

    fig,ax = plt.subplots(2,2)

    ax[0,0].plot(f[idx2], p[idx2])
    ax[0,0].plot(f[idx1], p[idx1])
    ax[0,0].set_xlabel(r'frequency [$\mu$Hz]')
    ax[0,0].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


    ax[0,1].plot(f[idx2], pds_w[idx2])
    ax[0,1].plot(f[idx1], pds_w[idx1])
    ax[0,1].plot(f[idx2],np.max(pds_w[idx1])*weight[idx2]/np.max(weight),color='k',ls='--')
    ax[0,1].set_xlabel(r'frequency [$\mu$Hz]')
    ax[0,1].set_ylabel(r'power [$ppm^2 / \mu$Hz]')


    ax[1,0].plot(lagvec2, acf2/acf2[1])
    ax[1,0].plot(lagvec1, acf1/acf1[1])
    ax[1,0].set_xlabel(r'frequency lag [$\mu$Hz]')
    ax[1,0].set_ylabel(f'ACF')
    for i in range(7):
        i+=1
        ax[1,0].vlines(x=dnu_peak*i,ymin=-1,ymax=1,ls='--',color='k')


    ax[1,1].plot(lagvec2, acf2_w/acf2_w[1])
    ax[1,1].plot(lagvec1, acf1_w/acf1_w[1])
    ax[1,1].set_xlabel(r'frequency lag [$\mu$Hz]')
    ax[1,1].set_ylabel(f'ACF')

    fig.tight_layout()
    plt.show()





'''

#loading data:
numax = 25.35 #muHz
dnu = 2.975 #muHz
df = pd.read_csv(folder_path + 'kplr009652971_kasoc-ts_llc_v1.dat',
                 skiprows=13,
                 delimiter=r"\s+").to_numpy()

time = df[:,0]
flux = df[:,1]
e_flux = df[:,2]

#reducing length of data:

N = len(time)
time,flux,e_flux = time[0:int(N/2)], flux[0:int(N/2)], e_flux[0:int(N/2)]

#Calculate psd:
psd = powerspectrum(time,flux,flux_err=e_flux,weighted=True)
f,p = psd.powerspectrum(scale='powerdensity')
idx_limit = (f<2*numax)
f, p0 = f[idx_limit], p[idx_limit]

#We try to remove smooth granulation background
p_bg = logmed_filter(f,p0, filter_width=0.2)
p = p0/p_bg

#Smoothing spectrum:
smo_width = 1
df = f[10]-f[9]
win = int(smo_width/df)
if win%2==0:
    win+=1

p_filt = filtering(p,win)
p_filt_ori = filtering(p0, win)

#Calculating Echelle
FWHM = numax /2
start = numax-FWHM
stop = numax+FWHM

A,B,EchelleVals = Echellediagram(p_filt,f,dnu,start,stop,no_repeat=1)
VMIN = np.min(EchelleVals)
VMAX = np.max(EchelleVals)



#plotting spectrum

fig, ax =   plt.subplots()
ax.plot(f,p0)
ax.plot(f,p_filt_ori)
ax.plot(f,p_bg)
plt.show()


fig, ax = plt.subplots()
ax.imshow(EchelleVals,aspect='auto',extent=[0,dnu,start,np.max(B)+0.5*dnu]
          ,norm = LogNorm(vmin=1.1,vmax=VMAX), interpolation='Gaussian',
          cmap=cm.gray_r,zorder=0)
plt.show()

'''





