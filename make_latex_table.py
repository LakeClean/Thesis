import latextable
from texttable import Texttable
import numpy as np
import make_table_of_target_info as mt

'''
rows = [["Mr\nXavier\nHuon",'', "Xav'"],
                     ["Mr\nBaptiste\nClement", 1, "Baby"],
                     ["Mme\nLouise\nBourgeau", 28, "Lou\n \nLoue"]]
names = np.array(["Name", "$Age$", "Nickname"])

def standard_table(names,rows,caption = '',label='',print_tab=True):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l", "r", "c"])
    table.set_cols_valign(["t", "m", "b"])

    table_output = []
    
    n = []
    for name in names:
        n.append(name)
    table_output.append(n)
    for row in rows:
        r = []
        for element in row:
            r.append(element)
        table_output.append(r)
    print(table_output)
            

    table.add_rows(table_output)
    if print_tab:
        print('-- Example 1: Basic --')
        print('Texttable Output:')
        print(table.draw())
        print('\nLatextable Output:')
        print(latextable.draw_latex(table,caption=caption,label=label))
    return latextable.draw_latex(table,caption=caption,label=label)
'''

        
        

def standard_table(names,columns,errors='', rounding=0,
                   caption = '',label='',print_tab=True):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["c"]*len(names))
    #table.set_cols_valign(["t", "m", "b"])

    table_output = []
    
    n = []
    for name in names:
        n.append(name)
    table_output.append(n)
    
    #Adding potential errors
    if len(errors) > 0:
        new_cols = [columns[0]]
        for i in zip(columns[1:],errors):
            new_col = []
            for element, err in zip(i[0],i[1]):
                new_col.append(f'{element} +/- {err}')
            new_cols.append(new_col)
    else:
        new_cols = columns
    print(new_cols)
    for row in zip(*new_cols):
        r = []
        for element in row:
            r.append(element)            
        table_output.append(r)
    print(table_output)
            

    table.add_rows(table_output)
    if print_tab:
        print('-- Example 1: Basic --')
        print('Texttable Output:')
        print(table.draw())
        print('\nLatextable Output:')
        print(latextable.draw_latex(table,caption=caption,label=label))
    return latextable.draw_latex(table,caption=caption,label=label)




tab = mt.get_table()


#Making table of relevant properties:
'''
names = ['Name', 'RA', 'DEC', 'V', 'SB type', 'Double-seismic']
IDs = tab['ID'].data
RAs = tab['RA'].data
DECs = tab['DEC'].data
mag = tab['Vmag'].data
SB_type = tab['SB_type'].data
d_seis = tab['double_seis'].data

standard_table(names,[IDs,RAs,DECs,mag,SB_type,d_seis])
'''

#Making table of orbital parameters from Gaia:
'''
names = ['Name','A', 'B', 'F', 'G']
IDs = tab['ID'].data
my_w = tab['w'].data
my_p = tab['p'].data
my_e = tab['e'].data
'''

#Making table of Thiele Innes parameters

names = ['Name','A', 'B', 'F', 'G']
IDs = tab['ID'].data
A = tab['G_ATI'].data
B = tab['G_BTI'].data
F = tab['G_FTI'].data
G = tab['G_GTI'].data
e_A = tab['e_G_ATI'].data
e_B = tab['e_G_BTI'].data
e_F = tab['e_G_FTI'].data
e_G = tab['e_G_GTI'].data

standard_table(names,[IDs,A,B,F,G],[e_A,e_B,e_F,e_G])























