import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import lmfit
from scipy.ndimage import gaussian_filter
import random as rd




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
    current_order = [-1]
    
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


        if event.key == 'a':
            current_order[0] += 1
            if len(points[0]) < current_order[0]+1:
                points[0].append(0)
                points[1].append(0)
                points[2].append(0)
                tags[0].append(0)
                tags[1].append(0)
                tags[2].append(0)
            
            print('Shifting the order to: ', current_order[0])
                
            return 'Shifting order'
        
        if event.key == 'd':
            current_order[0] -= 1
            print('Shifting the order to: ', current_order[0])
            return 'Shifting order'
        
        if event.key == 'z': k, color = 0, 'green'
        if event.key == 'x': k, color = 1, 'red'
        if event.key == 'v': k, color = 2, 'yellow'

        if event.key not in ['z', 'x', 'v']:
            print('You should press z, x, or v')
            return 'unknown key'
        
        
        
        if (type(x) == type(np.float64(1))) and (type(y) == type(np.float64(1))):
            removing_point = False



            if points[k][current_order[0]] != 0:
                tags[k][current_order[0]].remove() #removes plot from fig
                points[k][current_order[0]] = 0
                tags[k][current_order[0]] = 0
                removing_point = True
                canvas.draw()
  
            if removing_point==False:
                tag = ax.scatter(x,y,color=color,zorder=3)
                points[k][current_order[0]] = x
                tags[k][current_order[0]] = tag
                canvas.draw()

        
    cid = fig.canvas.mpl_connect('key_press_event', create_point)

   

    #### Clearing all points: ####
    def clear_points():
        for k in range(3):
            length_tag = len(tags[k])        
            for i in range(length_tag):
                tags[k][0].remove()
                tags[k].pop(0)
                points[k].pop(0)
        
        canvas.draw()
    clear_button = tk.Button(frame,  text='clear',command=clear_points)
    clear_button.pack(anchor='e')


    #### Quit button: ####
    def do_quit():
        root.destroy()
    quit_button =  tk.Button(frame, text='Quit', command=do_quit)
    quit_button.pack(anchor='e')

    root.mainloop()

    return np.array(points)

#Tests:
if False:
    i = np.linspace(-10,10,1000)
    #y = Gaussian(x,1,1,1) 
    j= [np.sin(x) + rd.uniform(0,1) for x in i]
    guess_points = simple_peak_bagging(i,j)
    print(guess_points)



