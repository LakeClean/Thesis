from astroquery.vizier import Vizier
import make_table_of_target_info as mt
import astropy.units as u

f = open('/home/lakeclean/Documents/speciale/NOT/Target_names_and_info.txt').read().split('\n')[36:51]

for line in f:
    line = line.split('\t')
    


tab = mt.get_table()
t_IDs = tab['ID'].data
IDs = tab['Gaia_ID'].data


for i in range(len(IDs)):
    result = Vizier(row_limit=10,columns = ['Vmag', 'RAJ2000',
                                      'DEJ2000']).query_object(
                                         f'Gaia DR3 {IDs[i]}',
                                     catalog=['I/322'],
                                       radius=5*u.arcsec)
    
    print(result[0]['Vmag'].data[0], 'Vmag',t_IDs[i])
    #mt.add_value(result[0]['Vmag'].data[0], 'Vmag',t_IDs[i])
    #mt.add_value(result[0]['RAJ2000'].data[0], 'RA',t_IDs[i])
    #mt.add_value(result[0]['DEJ2000'].data[0], 'DEC',t_IDs[i])
    

