def get_goettingen():

    """
    import read_goettingen as gg 
    wn, sun, tel, wave = gg.get_goettingen()
    """

    import astropy.io.fits as fits
    import numpy as np

    directory  = '~/Speciale/data/templates/goettingen/'
    atlas_file = directory + 'goettingen_sun_visual.fits'
    atlas      = fits.getdata( atlas_file, 1 )
    wavenum0   = np.flip( atlas[ 'wavenum' ] )
    flux0      = np.flip( atlas[ 'flux' ] )
    telluric0  = np.flip( atlas[ 'tel' ] )
    wave_air0  = np.flip( atlas[ 'wav_air' ] )

    wavenum    = wavenum0[ 0, :, 0 ]
    flux       = flux0[ 0, :, 0 ]
    telluric   = telluric0[ 0, :, 0 ]
    wave_air   = wave_air0[ 0, :, 0 ]

    return wavenum, flux, telluric, wave_air
