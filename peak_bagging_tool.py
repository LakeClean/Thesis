import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import lmfit
from scipy.ndimage import gaussian_filter
import random as rd


def Gaussian(x,std,mu,floor):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x-mu)**2 / (std**2) ) + floor

def Gaussian_res(params,x,y):
    std = params['std'].value
    mu = params['mu'].value
    floor = params['floor'].value
    res = y - Gaussian(x,std,mu,floor)
    return res

def mode(x,eps,H,gam,nu,const):
    return eps*H / (1 + 4/gam**2 * (x - nu)**2) + const

def mode_res(params,x,y):
    eps = params['eps'].value
    H = params['H'].value
    gam = params['gam'].value
    nu = params['nu'].value
    const = params['const'].value
    res = y - mode(x,eps,H,gam,nu,const)
    return res  






def simple_peak_bagging(x_val,y_val,region=[],fitter='mode'):
    x,y = x_val, y_val

    #Creating canvas
    root = tk.Tk()
    root.geometry('800x700')
    root.title('Bagging peaks')

    #matplotlib figure
    fig, ax = plt.subplots()
    original_power = ax.plot(x,y,color='b',zorder=1,alpha=0.2)
    smoothed_power = ax.plot(x,y,color='b',zorder=2)

    #Tkinter application:
    frame = tk.Frame(root)
    label = tk.Label(text='peakbagging')
    label.pack()

    # Create a canvas widget
    canvas=FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack()

    # Toolbar:
    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(anchor='w', fill=tk.X)
    frame.pack()

    #### Smooth power spectrum: ####
    sigma = 0
    def smooth():
        USER_INP_sigma = tk.simpledialog.askstring(title = 'Smoothing',
                                                 prompt = 'sigma smoothin int:')
        try:
            sigma = int(USER_INP_sigma)
        except:
            print('input was given wrong')
            return 0

        smoothed = gaussian_filter(y[:], sigma=sigma)
        smoothed_power[0].set_ydata(smoothed)
        canvas.draw()
        
    smooth_button = tk.Button(frame,  text='smooth',command=smooth)
    smooth_button.pack(anchor='e')

    #Creating and deleting singular points:

    points = [[],[],[]]
    tags  = [[],[],[]]
    def create_point(event):
        axinfo = str(ax.viewLim)[4:].split(',')
        axx0 = float(axinfo[0][4:])
        axy0 = float(axinfo[1][4:])
        axx1 = float(axinfo[2][4:])
        axy1 = float(axinfo[3][4:-1])
        xlimit = (axx1 - axx0)/50
        ylimit = (axy1 - axy0)/50

        x=event.xdata
        y=event.ydata

        if event.key == 'z': k, color = 0, 'green'
        if event.key == 'x': k, color = 1, 'red'
        if event.key == 'v': k, color = 2, 'yellow'

        if event.key not in ['z', 'x', 'v']:
            print('You should press z, x, or v')
            return 'unknown key'
        
        

        if (type(x) == type(np.float64(1))) and (type(y) == type(np.float64(1))):
            removing_point = True
            for i,point in enumerate(points[k]):
               xdistance = abs(point[0] - x)
               ydistance = abs(point[1]- y)
               if ydistance < ylimit:
                   if xdistance < xlimit:
                       tags[k][i].remove()
                       points[k].pop(i)
                       tags[k].pop(i)
                       removing_point = False
                       canvas.draw()

            if ((x,y) not in points[k]) and (removing_point==True):
                tag = ax.scatter(x,y,color=color,zorder=3)
                points[k].append((x,y))
                tags[k].append(tag)
                canvas.draw()
        
    cid = fig.canvas.mpl_connect('key_press_event', create_point)

    #### Fitting Gaussian to peaks: ####

    fit_tags = []
    gauss_params = [[],[],[]]
    def fit_Gauss():

        if len(region) == 0:
            USER_INP_z = tk.simpledialog.askstring(title = 'region z: ',
                                                 prompt = 'input int: ')
            USER_INP_x = tk.simpledialog.askstring(title = 'region x: ',
                                                 prompt = 'input int: ')
            USER_INP_v = tk.simpledialog.askstring(title = 'region v: ',
                                                 prompt = 'input int: ')
            try:
                region = [float(USER_INP_z),float(USER_INP_x),
                          float(USER_INP_v)]
            except:
                print('You entered region values that is not valid!')
                return 0,0,0
           

        for k in range(3):
            for point in points[k]:
                x_idx = (( point[0] - region[k] < x ) & ( point[0] + region[k] > x ) )
                x_lim = x[x_idx]
                y_lim = y[x_idx]

                ## starting guesses: ##
                idx = np.argmin(abs(y_lim - point[1]))
                mu1 = x_lim[idx]
                
                params = lmfit.Parameters()
                params.add('std',value=1)
                params.add('mu', value=mu1)
                params.add('floor',value=1)

                fit = lmfit.minimize(Gaussian_res, params, args=(x_lim,y_lim),
                                     xtol=1.e-8,ftol=1.e-8,max_nfev=500)
                print(lmfit.fit_report(fit,show_correl=False))
                
                std = fit.params['std'].value
                mu = fit.params['mu'].value
                floor = fit.params['floor'].value

                gauss_params[k].append([std,mu,floor])
                
                #plotting:
                fit_tag = ax.plot(x_lim,Gaussian(x_lim,std,mu,floor),
                                  color='green',zorder=3)
                fit_tags.append(fit_tag[0])
                canvas.draw()

    mode_params = [[],[],[]]
    def fit_mode():

        if sigma != 0:
            smoothed = gaussian_filter(y[:], sigma=sigma)
        else:
            smoothed = y[:]

        smoothed = np.array(smoothed)

        
        if len(region) == 0:
            USER_INP_z = tk.simpledialog.askstring(title = 'region z: ',
                                                 prompt = 'input int: ')
            USER_INP_x = tk.simpledialog.askstring(title = 'region x: ',
                                                 prompt = 'input int: ')
            USER_INP_c = tk.simpledialog.askstring(title = 'region c: ',
                                                 prompt = 'input int: ')
            try:
                new_region = [float(USER_INP_z),float(USER_INP_x),
                              float(USER_INP_c)]
            except:
                print('You entered region values that is not valid!')
                return 0,0,0
            
        for k in range(3):
            for point in points[k]: 
                x_idx = (( point[0] - new_region[k] < x ) &
                         ( point[0] + new_region[k] > x ) )
                x_lim = x[x_idx]
                y_lim = smoothed[x_idx]
                
                ## starting guesses: ##
                idx = np.argmin(abs(y_lim - point[1]))
                mu1 = x_lim[idx]
                
                params = lmfit.Parameters()
                params.add('eps',value=1)
                params.add('H', value=3)
                params.add('gam',value=10)
                params.add('nu',value=point[0])
                params.add('const',value=0)

                fit = lmfit.minimize(mode_res, params, args=(x_lim,y_lim),
                                     xtol=1.e-8,ftol=1.e-8,max_nfev=500)
                print(lmfit.fit_report(fit,show_correl=False))
                
                eps = fit.params['eps'].value
                H = fit.params['H'].value
                gam = fit.params['gam'].value
                nu = fit.params['nu'].value
                const = fit.params['const'].value

                mode_params[k].append([eps,H,gam,nu,const])
                
                #plotting:
                fit_tag = ax.plot(x_lim,mode(x_lim,eps,H,gam,nu,const),
                                  color='red',zorder=3)
                fit_tags.append(fit_tag[0])
                canvas.draw()


    Gauss_button = tk.Button(frame,  text='Fit Gaussian',
                                 command=fit_Gauss)
    Gauss_button.pack(anchor='e')
        
    mode_button = tk.Button(frame,  text='Fit mode',
                                 command=fit_mode)
    mode_button.pack(anchor='e')
        


    #### Clearing all points: ####
    def clear_points():
        for k in range(3):
            length_tag = len(tags[k])        
            for i in range(length_tag):
                tags[k][0].remove()
                tags[k].pop(0)
                points[k].pop(0)

            length_fit_tag = len(fit_tags)
            for i in range(length_fit_tag):
                fit_tags[0].remove()
                fit_tags.pop(0)
                
            length_mode_params = len(mode_params[k])
            for i in range(length_mode_params):
                mode_params[k].pop(0)

            length_gauss_params = len(gauss_params[k])
            for i in range(length_gauss_params):
                gauss_params[k].pop(0)
        
        canvas.draw()
    clear_button = tk.Button(frame,  text='clear',command=clear_points)
    clear_button.pack(anchor='e')


    #### Quit button: ####
    def do_quit():
        root.destroy()
    quit_button =  tk.Button(frame, text='Quit', command=do_quit)
    quit_button.pack(anchor='e')

    root.mainloop()

    return points, gauss_params, mode_params

#Tests:
if False:
    i = np.linspace(-10,10,1000)
    #y = Gaussian(x,1,1,1) 
    j= [np.sin(x) + rd.uniform(0,1) for x in i]
    print(simple_peak_bagging(i,j))

