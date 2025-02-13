#Directory for estimating seismic parameters for stars with single star signal.

#Order:

    - find_ACF
    - find_peaks
    - find_error_in_peaks
    - find_numax
    - plot_echelle

# The following targets have been run:
    - KIC10454113: all the way to find_error_in_peaks
    - KIC9693187: find peaks
    - KIC9025370: find peaks
    - KIC12317678: Has several errornus error estimations
    - KIC4914923: find peaks

Should probably be run again with higher burnin
and with non tophat(uniform) prior.
Something like the jefferys prior.
find_peaks could maybe also benefit from being run again now that bounds
have been set for the least squares fit.
