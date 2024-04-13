# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:47:56 2023

@author: garub
"""

import file as f
import figure
import data
import pandas as pd
import scipy.signal as ss
import plotly.io as pio
import pybaselines.spline as py
import os

pio.renderers.default = 'jpg'

root = "C:/Users/garub/OneDrive/Desktop/lps-data/"
indir = root + '525_data1/'
outdir = root + '/525_data2/'

def truncate_spectrum(x, y, start, end):
    """
    Truncates the Raman spectrum on the Raman shift axis.

    Args:
        x (array-like): Raman shift values.
        y (array-like): Intensity values.
        start (float): Start value of the truncated range.
        end (float): End value of the truncated range.

    Returns:
        tuple: Truncated Raman shift values and corresponding intensity values.
    """
    mask = (x >= start) & (x <= end)
    truncated_x = x[mask]
    truncated_y = y[mask]
    return truncated_x, truncated_y

if not os.path.exists(outdir):
    os.makedirs(outdir)

filelist = f.names.find(indir, contains='')

def process(file):
    # 1)Extract data + name from file
    df = pd.read_csv(indir + file, header=None)
    x, y0 = data.extract_col(df, 0), data.extract_col(df, 1)
    fullname = f.names.edit(file, '.', 0)

    # 2) Fix Spectra: remove spikes from cosmic rays
    try:
        y = data.zap.fix(y0)
    except IndexError:  # Bias IndexError for when spike is detected at edges of datapp
        y = y0
        pass

    # 3)Smooth Intensity
    iZS = ss.savgol_filter(y, 9, 2)

    # 4) Apply baseline
    b_out = py.pspline_asls(iZS)[0]
    iZSB = y - b_out

    # 5) Processing Figure
    fig = figure.process(x, y,iZSB, b_out, fullname)

    # 6) Returns
    return x, iZSB, fullname, fig, y

if __name__ == '__main__':
    figdir = outdir + 'fig/'
    
    # Prompt user for truncation range
    start_value = float(input("Enter the start value for truncation: "))
    end_value = float(input("Enter the end value for truncation: "))
    
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    
    for file in filelist:
        X, Y, Name, Fig, Raw = process(file)
        
        # Truncate the spectrum
        X_truncated, Y_truncated = truncate_spectrum(X, Y, start_value, end_value)
        
        # Save truncated data to CSV
        Df = pd.DataFrame(list(zip(X_truncated, Y_truncated)), columns=['raman shift', 'intensity'])
        Df.to_csv('%s%s.csv' % (outdir, Name), index=None)
        
        # Display and save the figure
        #Fig.show()
        #figure.save.static(Fig, figdir, Name)
        
        print() 
    
    print('End.')
