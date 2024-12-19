# Thesis
Repository for my thesis work

#Pipeline for NOT data:
- The ordered files are downloaded and called reduced*.zip. Should be in Downloads      |By Hand|
- Note the date of the oldest of these for later use in analyse_spectra                 |By Hand|
- They are unzipped and moved to dir 'initial_data'                                     |unzip.py|
- NOT_order_file_log.txt is updated. This need pm_px.txt and intial_data to be updated. |read_NOT_header.py|
- The folders for the data made in 'analyse_ordered_spectra.py' are made                |make_data_folders.py|
- spectra_log_h_readable.txt is updated for new dates                                   |h_readable_log.py|
- spectra_log_h_readable.txt is updated for peaks of SB2's                              |By Hand|
- spectra are analysed and bf found and fitted to.                                      |analayse_ordered_spectra.py|
- Rv is determined and figures are made.                                                |make_rv_figures.py|

#Pipeline for TNG (HARPS):
- The merged files are downloaded and manually written path in script.                  |By Hand|
- Note the date of the oldest of these for later use in analyse_spectra                 |By Hand|
- They are untared and moved to dir 'initial_data'                                      |untar.py|
- TNG_merged_file_log.txt is updated. This need pm_px.txt and intial_data to be updated.|read_TNG_header.py|
- The folders for the data made in 'analyse_ordered_spectra.py' are made                |make_data_folders.py|
- spectra_log_h_readable.txt is updated for new dates                                   |h_readable_log.py|
- spectra_log_h_readable.txt is updated for peaks of SB2's                              |By Hand|
- spectra are analysed and bf found and fitted to.                                      |analayse_merged_spectra.py|
- Rv is determined and figures are made.                                                |make_rv_figures.py|


#Pipeline for (KECK):
- The ordered files are downloaded                                                      |By Hand|
- Note the date of the oldest of these for later use in analyse_spectra                 |By Hand|
- They are untared and moved to dir 'initial_data/KECK/targetname'                      |untar.py|
- KECK_order_file_log.txt is updated. This need pm_px.txt and intial_data to be updated.|read_KECK_header.py|
- The folders for the data made in 'analyse_ordered_spectra.py' are made                |make_data_folders.py|
- spectra_log_h_readable.txt is updated for new dates                                   |h_readable_log.py|
- spectra_log_h_readable.txt is updated for peaks of SB2's                              |By Hand|
- spectra are analysed and bf found and fitted to.                                      |analayse_KECK_spectra.py|
- Rv is determined and figures are made.                                                |make_rv_figures.py|


#Pipeline for (ESpaDOns): 
- The ordered files are downloaded                                                      |By Hand|
- They are untared and moved to dir 'initial_data/ESpaDOns/targetname'                  |untar.py|
- ESpaDOns_order_file_log.txt is updated.                                               |read_ESpaDOns_header.py|
- The folders for the data made in 'analyse_ordered_spectra.py' are made                |make_data_folders.py|
- spectra_log_h_readable.txt is updated for new dates                                   |h_readable_log.py|
- spectra_log_h_readable.txt is updated for peaks of SB2's                              |By Hand|
- spectra are analysed and bf found and fitted to.                                      |analayse_ESpaDOns_spectra.py|
- Rv is determined and figures are made.                                                |make_rv_figures.py|
note that we left behind:
/home/lakeclean/Documents/speciale/initial_data/ESpaDOns/2016-09-19/2005112p.fits
due to the peculiar nature of the fits file.


#Unfixed bugs:
 - The pipeline needs the TCSTGT name to be the specific one either with '-' or without
 - The SEQID has to be science
 - h_readable_log.py messes up the order of the files

  




