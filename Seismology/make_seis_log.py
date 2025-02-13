import glob
import pandas as pd

master_path = '/usr/users/au662080'

folders = glob.glob(f'{master_path}/Speciale/data/Seismology/initial_data/*')
file_paths = []
for folder in folders:
    files = glob.glob(folder + '/*')
    for file in files:
        file_paths.append(file)



IDs = ['KIC9693187','KIC4260884', 'KIC10454113', 'KIC9652971', 'KIC4457331',
       'KIC4914923', 'KIC12317678', 'KIC9025370','EPIC212617037',
       'EPIC246696804','EPIC230748783','EPIC249570007','EPIC212617037',
       'EPIC236224056']

data_type = ['pow','dat','pow','dat','dat','pow','pow','pow','pow',
             'pow','pow','pow','dat','pow']


numax1_guess = [1000,50.41,2250,25.35,76.48,1800,1200,3000,
                1000,1000,1000,1000,1000,1000]
numax2_guess = [1000, 120,2250,40,110,1800,1200,3000,
                1000,1000,1000,1000,1000,1000]

data = [IDs,data_type,numax1_guess,numax2_guess,file_paths]    

out_dict = {}
header = ['ID', 'data_type', 'numax1_guess', 'numax2_guess','data_path']
for i in range(len(header)):
    out_dict[header[i]] = data[i]

out_path = f'{master_path}/Speciale/data/Seismology/analysis/log_file.txt'
out_df = pd.DataFrame(out_dict)
out_df.to_csv(out_path,index=False)
    
    
















    
