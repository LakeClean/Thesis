import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob
import pandas as pd




#importting info:
path = '/home/lakeclean/Documents/speciale/target_analysis/'
def plot_bf(ID,date=None,bin_range=None):
    
    if type(date) == int:
        directory = glob.glob(path + ID + '/*')
        date = directory[date][len(path + ID + '/'):]
        
    elif date == None:
        directory = glob.glob(path + ID + '/*')
        date = directory[0][len(path + ID + '/'):]
        
    
    bf_path = path + ID + '/' + date + '/data/'

    def sorter(x):
        '''
        key for sorter
        '''
        try:
            result = int(x[len(bf_path)+3:len(bf_path)+5])
        except:
            result = int(x[len(bf_path)+3:len(bf_path)+4])
        return result
        
    files = glob.glob(bf_path + 'bin*_broadening_function.txt')
    files.sort(key=sorter)
    bin_i_data = []
    for file in files:
        df = pd.read_csv(file)
        bin_i_data.append(df.to_numpy())
        
    # Define initial parameters
    init_bin = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = ax.plot(bin_i_data[0][:,0], bin_i_data[0][:,1], lw=2)
    ax.set_ylim(-2,2)
    ax.set_xlabel('rv [km/s]')
    ax.set_ylabel('broadening function')
    ax.set_title(f'broadening function, {ID}, {date}')
    # adjust the main plot to make room for the slider
    fig.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    bin_slider = Slider(
        ax=axfreq,
        label='bin#',
        valmin=0,
        valmax=len(files)-1,
        valinit=init_bin,
        valstep=1
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(bin_i_data[bin_slider.val][:,1])
        line.set_xdata(bin_i_data[bin_slider.val][:,0])
        fig.canvas.draw_idle()


    # register the update function with each slider
    bin_slider.on_changed(update)

    plt.show()


def plot_norm(ID,date=None,bin_range=None):
    pass

def plot_ccf(ID,date=None,bin_range=None):

    if type(date) == int:
        directory = glob.glob(path + ID + '/*')
        date = directory[date][len(path + ID + '/'):]
        
    elif date == None:
        directory = glob.glob(path + ID + '/*')
        date = directory[0][len(path + ID + '/'):]
        
    ccf_path = path + ID + '/' + date + '/data/'

    def sorter(x):
        '''
        key for sorter
        '''
        try:
            result = int(x[len(ccf_path)+3:len(ccf_path)+5])
        except:
            result = int(x[len(ccf_path)+3:len(ccf_path)+4])
        return result
        
    files = glob.glob(ccf_path + 'bin*_cross_correlation.txt')
    files.sort(key=sorter)
    bin_i_data = []
    for file in files:
        df = pd.read_csv(file)
        bin_i_data.append(df.to_numpy())
        
    # Define initial parameters
    init_bin = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = ax.plot(bin_i_data[0][:,0], bin_i_data[0][:,1], lw=2)
    ax.set_ylim(-2,3)
    ax.set_xlabel('rv [km/s]')
    ax.set_ylabel('cross correlation')
    ax.set_title(f'cross correlation, {ID}, {date}')

    # adjust the main plot to make room for the slider
    fig.subplots_adjust(bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    bin_slider = Slider(
        ax=axfreq,
        label='bin#',
        valmin=0,
        valmax=len(files)-1,
        valinit=init_bin,
        valstep=1
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(bin_i_data[bin_slider.val][:,1])
        line.set_xdata(bin_i_data[bin_slider.val][:,0])
        fig.canvas.draw_idle()


    # register the update function with each slider
    bin_slider.on_changed(update)

    plt.show()
plot_ccf('KIC-9693187',1) 
