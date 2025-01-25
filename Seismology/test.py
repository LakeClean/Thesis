import matplotlib.pyplot as plt
import numpy as np
from ps import powerspectrum



def Gauss(x,a,b,c):
    return a*np.exp(-0.5*((x-b)/c)**2)


ts = np.linspace(-10,10,5000)


ys = []
for i in ts:
    prob = np.random.uniform(0,1)
    x = np.random.uniform(-10,10)

    while prob > Gauss(x,1,0,2):
        prob = np.random.uniform(0,1)
        x = np.random.uniform(-10,10)

    ys.append(x)


ys = np.array(ys)



PDS = powerspectrum(ts,ys)
f, p0 = PDS.powerspectrum(scale='powerdensity')

fig, ax = plt.subplots()
ax.scatter(ts,ys,s=0.2)

fig, ax = plt.subplots()
ax.scatter(f,p0,s=0.2)

plt.show()

bins = range(-10,10,1)
dist = [0]
for bin1,bin2 in zip(bins[0:],bins[1:]):
    summed = 0
    for y in ys:
        if (y>bin1) and (y<bin2):
            summed += 1

    dist.append(summed)
            
plt.plot(bins,dist)
plt.show()
        
    
