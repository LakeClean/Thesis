import make_table_of_target_info as mt
import matplotlib.pyplot as plt
import numpy as np
import convert_to_Campbell as ctc

tab = mt.get_table()
fig,ax = plt.subplots(1,4,sharey = True)


IDs = tab['ID'].data

#mass
M1s = tab['M1'].data
e_M1s = tab['e_M1'].data
M1_litts = tab['M1_seis_litt'].data

e_M1_litts = tab['e_M1_seis_litt'].data

#eccentricity
es = tab['e'].data
e_litts = tab['e_seis_litt'].data
e_es = tab['e_e'].data
e_e_litts = tab['e_e_seis_litt'].data

#period:

ps = tab['p'].data
p_litts = tab['p_seis_litt'].data
e_ps = tab['e_p'].data
e_p_litts = tab['e_p_seis_litt'].data

#argument of periastron:

ws = tab['w'].data
#w_litts = tab['w_seis_litt'].data
e_ws = tab['e_w'].data
#e_w_litts = tab['e_w_seis_litt'].data


#Thiele Innes parameters:

A = tab['G_ATI'].data
B = tab['G_BTI'].data
F = tab['G_FTI'].data
G = tab['G_GTI'].data
e_A = tab['e_G_ATI'].data
e_B = tab['e_G_BTI'].data
e_F = tab['e_G_FTI'].data
e_G = tab['e_G_GTI'].data






SB1 = ['KIC4914923']
SB2 = ['KIC9025370', 'KIC12317678', 'KIC9693187']
#both = ['KIC-9025370', 'KIC-12317678', 'KIC-9693187','KIC-4914923']

both = {'KIC9025370':1, 'KIC12317678':2, 'KIC9693187':3,'KIC4914923':4}

for i in range(len(IDs)):

    if IDs[i] in SB2:
        ax[0].errorbar(M1s[i], both[IDs[i]], xerr=e_M1s[i],capsize=2,color='blue')
        
        if np.isnan(e_M1_litts[i]):
            ax[0].scatter(M1_litts[i], both[IDs[i]]+0.1,s=5,color='red')
        else:
            ax[0].errorbar(M1_litts[i], both[IDs[i]]+0.1, xerr=e_M1_litts[i],capsize=2,color='red')

    if IDs[i] in both.keys():
        #Thiele-Innes:
        G_a,G_Omega,G_w,G_i ,e_G_a,e_G_Omega,e_G_w,e_G_i = ctc.find_Campbell(A[i],B[i],
                                                                             F[i],G[i],
                                                                             e_A[i],e_B[i],
                                                                             e_F[i],e_G[i])

        
        
        #period:
        ax[1].errorbar(ps[i], both[IDs[i]], xerr=e_ps[i],capsize=2,color='blue')
        ax[1].errorbar(p_litts[i], both[IDs[i]]+0.1, xerr=e_p_litts[i],capsize=2,color='red')

        #eccentricity:
        ax[2].errorbar(es[i], both[IDs[i]], xerr=e_es[i],capsize=2,color='blue')
        ax[2].errorbar(e_litts[i], both[IDs[i]]+0.1, xerr=e_e_litts[i],capsize=2,color='red')
        

        #Argument of periastron:
        ax[3].errorbar(ws[i], both[IDs[i]], xerr=e_ws[i],capsize=2,color='blue')
        ax[3].errorbar(G_w, both[IDs[i]]+0.1, xerr=e_G_w,capsize=2,color='green')



ax[0].set_xlabel('M1 [M_sun]')
ax[1].set_xlabel('P [days]')
ax[2].set_xlabel('e')
ax[3].set_xlabel('w [deg]')


ax[0].plot(0,0,color='red',label='Litterature')
ax[0].plot(0,0,color='blue',label='This work')
ax[0].plot(0,0,color='green',label='Gaia')
ax[0].legend()

#labels = ['KIC-9025370', 'KIC-12317678', 'KIC-9693187','KIC-4914923']
ax[0].set_ylim(0.8,4.2)
ax[0].set_yticks(ticks = [1,2,3,4], labels=both.keys())

#line1 = set_label('This work')
#line2 = set_label('Litterature')
fig.tight_layout()
fig.subplots_adjust(wspace=0)
plt.show()


