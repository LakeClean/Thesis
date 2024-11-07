import barycorrpy as bc
import astropy.io.fits as pyfits


#header = pyfits.getheader('FIHd010095_step011_merge.fits')

# Observation of Tau Ceti taken from CTIO on JD 2458000,2458010,2458020.
        # Observatory and stellar parameters entered by user.
        # Use DE405 ephemeris
'''
EPIC-246696804, science,
2024-04-01T20:24:56.782, 300.0,
-26.0645537951034, F4 HiRes, 2062, 71.6425899394748, 14.5399433379132,
28.757222222, -17.885, 2465.5, 10.77, 15.3, 9.89, 2460402.350657199, 0
'''
ra=71.6426477738346
dec=14.5399867165479
obsname=''
lat= 28.757222222
longi=-17.885
alt=2465.5
epoch = 2460402.349930903
pmra = 10.77
pmdec = 15.3
px = 9.89
rv = 0.0
zmeas=0.0
JDUTC=epoch # Can also enter JDUTC as float instead of Astropy Time Object
#ephemeris='https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp'

result3=bc.get_BC_vel(JDUTC=JDUTC, ra=ra, dec=dec, obsname=obsname, lat=lat, longi=longi, alt=alt,
                   pmra=pmra,pmdec=pmdec, px=px, leap_update=True)
print(result3[0])
