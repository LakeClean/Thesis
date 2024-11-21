import latextable
from texttable import Texttable
import numpy as np

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

def standard_table(names,columns,caption = '',label='',print_tab=True):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_deco(Texttable.HEADER)
    #table.set_cols_align(["l", "r", "c"])
    #table.set_cols_valign(["t", "m", "b"])

    table_output = []
    
    n = []
    for name in names:
        n.append(name)
    table_output.append(n)
  
    for row in zip(*columns):
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


standard_table(names,[a,b,c,d])








