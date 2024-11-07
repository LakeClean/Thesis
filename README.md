# Thesis
Repository for my thesis work


#Pipeline is as follows:
- The ordered files are downloaded and called reduced*.zip. Should be in Downloads  |By Hand|
- They are unzipped and moved to dir 'initial_data'                                 |unzip.py|
- order_file_log.txt is updated. This need pm_px.txt and intial_data to be updated. |read_header.py|
- The folders for the data made in 'analyse_ordered_spectra.py' are made            |purge_target_analysis.py|
- spectra_log_h_readable.txt is updated for new dates                               |h_readable_log.py|
- spectra_log_h_readable.txt is updated for peaks of SB2's                          |By Hand|
- spectra are analysed and bf found and fitted to.                                  |analayse_ordered_spectra.py|
- Rv is determined and figures are made.                                            |make_rv_figures.py|

  




