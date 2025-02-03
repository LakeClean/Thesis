import numpy as np
from astroquery.vizier import Vizier
from ophobningslov import *
import sympy as sp
import make_table_of_target_info as mt
import astropy.units as u
import astropy.coordinates as coord

tab = mt.get_table()
IDs, G_IDs = tab['ID'].data, tab['Gaia_ID'].data
RAs, DECs = tab['RA'].data, tab['DEC'].data

def find_parameter(ID, parameters=[]):
    #Finding Gaia ID:
    idx = (IDs == ID)
    G_ID = G_IDs[idx]
    RA = RAs[idx]
    DEC = DECs[idx]
    
    #Catalogs:
    TI = ['I/357/tboasb1c','I/357/tbooc']
    seis = ['J/A+A/674/A106/table1','J/ApJ/835/173/table3',
            'J/ApJS/210/1/table1','J/ApJS/236/42/giants',
            'J/ApJS/233/23/table3','J/ApJ/844/102/table1']
    gaia = ['I/345/gaia2']
    apogee = ['III/286/allvis','III/284/allstars','J/A+A/450/735/table2',
              'III/286/catalog', 'III/284/allvis', 'J/ApJ/879/69']
    spec = ['IV/34/epic','J/A+A/530/A138/catalog']
    MNRAS = ['J/MNRAS/481/3244/marvels', 'J/MNRAS/434/1422/table3',
             'J/MNRAS/423/122/table3']
    Mathur = ['J/ApJS/229/30/catalog']
    Kepler_team = ['V/133/kic']
    LAMOST = ['J/A+A/594/A39/tablea3','J/ApJS/264/17/table1',
              'J/A+A/594/A39/tablea3']
    Huber = ['J/ApJ/844/102/table2']
    
    

    catalogs = TI + seis + gaia + apogee + spec + MNRAS
    catalogs += Mathur + Kepler_team + LAMOST + Huber

    if len(parameters) == 0:
    
        result = Vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC,
                                            unit=(u.deg, u.deg),
                                                frame='icrs'),
                                radius=5*u.arcsec,
                                catalog=catalogs)
    else:
        vizier = Vizier(columns = parameters)
        result = vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC,
                                            unit=(u.deg, u.deg),
                                                frame='icrs'),
                                radius=5*u.arcsec,
                                catalog=catalogs)
    return result


#Thiele Innes:
# 'ATI', 'BTI', 'FTI', 'GTI', 'CTI', 'HTI'
# Period

result = find_parameter('KIC4914923')
#print(result[0])


# 4260884, 9652971
RGBs = ['KIC4457331', 'KIC4260884', 'KIC9652971']
MSs = ['KIC4914923', 'KIC9025370', 'KIC10454113',
       'KIC12317678','KIC4260884', 'KIC9652971']
for j in MSs:
    break

    print(j)
    for i in find_parameter(j):
        try:
            print(i['numax'])
        except:
            pass
        try:
            print(i['__dnu'])
        except:
            pass
        try:
            print(i['Dnu'])
        except:
            pass




'''  
catalog = ['I/357/tboasb1c','I/357/tbooc','J/A+A/674/A106/table1']

GDR3 = ['2107491287660520832','2126516412237386880','2131620306552653312','2101240083023021952',
        '2135483028339642368', '6572992562249889280']


result = Vizier(row_limit=3).query_constraints(catalog=['I/357/tboasb1c',
                                                        'I/357/tbooc',
                                                        'J/A+A/674/A106/table1'],
                                               Source=GDR3)

result = Vizier.query_region(coord.SkyCoord(ra=299.590, dec=35.201,
                                            unit=(u.deg, u.deg),frame='icrs'),
                        radius=10*u.arcmin,
                        catalog=catalog)
print(result)


A = float(result[0]['ATI'])
B = float(result[0]['BTI'])
F = float(result[0]['FTI'])
G = float(result[0]['GTI'])

print(A,B,F,G)
'''
