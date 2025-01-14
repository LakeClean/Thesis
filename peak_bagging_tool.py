import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import lmfit


def Gaussian(x,std,mu,floor):
    return 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x-mu)**2 / (std**2) ) + floor

def Gaussian_res(params,x,y):
    std = params['std'].value
    mu = params['mu'].value
    floor = params['floor'].value
    res = y - Gaussian(x,std,mu,floor)
    return res
    

def simple_peak_bagging(x,y,region=0):

    #Creating canvas
    root = tk.Tk()
    root.geometry('800x700')
    root.title('Bagging peaks')

    #matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(x,y)

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

    #Creating and deleting singular points:
    points = []
    tags  = []
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

        if (type(x) == type(np.float64(1))) and (type(y) == type(np.float64(1))):
            removing_point = True
            for i,point in enumerate(points):
               xdistance = abs(point[0] - x)
               ydistance = abs(point[1]- y)
               if ydistance < ylimit:
                   if xdistance < xlimit:
                       tags[i].remove()
                       points.pop(i)
                       tags.pop(i)
                       removing_point = False
                       canvas.draw()

            if ((x,y) not in points) and (removing_point==True):
                tag = ax.scatter(x,y)
                points.append((x,y))
                tags.append(tag)
                canvas.draw()
        
    cid = fig.canvas.mpl_connect('key_press_event', create_point)

    #### Fitting Gaussian to peaks: ####

    fit_tags = []
    def fit_Gauss():
        global x
        global y
        for point in points:
            if region != 0:
                x_idx = (( point[0] - region < x ) & ( point[0] + region > x ) )
                x_lim = x[x_idx]
                y_lim = y[x_idx]
            else:
                x_lim = x[:]
                y_lim = y[:]

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
            
            #plotting:
            fit_tag = ax.plot(x_lim,Gaussian(x_lim,std,mu,floor))
            fit_tags.append(fit_tag[0])
            canvas.draw()

    Gauss_button = tk.Button(frame,  text='Fit Gaussian',
                             command=fit_Gauss)
    Gauss_button.pack(anchor='e')


    #### Clearing all points: ####
    def clear_points():
        length_tag = len(tags)
        length_fit_tag = len(fit_tags)
        for i in range(length_tag):
            tags[0].remove()
            tags.pop(0)
            points.pop(0)
            
        for i in range(length_fit_tag):
            fit_tags[0].remove()
            fit_tags.pop(0)

        
        canvas.draw()
    clear_button = tk.Button(frame,  text='clear',command=clear_points)
    clear_button.pack(anchor='e')

    #### Quit button: ####
    def do_quit():
        root.destroy()
    quit_button =  tk.Button(frame, text='Quit', command=do_quit)
    quit_button.pack(anchor='e')

    root.mainloop()

    return points

#Tests:
'''
x = np.linspace(-10,10,1000)
#y = Gaussian(x,1,1,1) 
y = np.sin(x)
simple_peak_bagging(x,y,region=2)
'''
