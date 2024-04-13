#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:34:58 2022

@author: dunn
"""
import plotly. graph_objects as go
#import plotly.express as px
from plotly.subplots import make_subplots
#from plotly.offline import plot
#from scipy.signal import find_peaks
import file as f

class save:
    #suffix (fig type) does include '.pdf' and ',=.svg'
    def static(fig, path, name, suffix='jpeg'):
        f.check_path(path) #from 'file.py'
        fig.write_image('%s%s.%s'%(path,name,suffix))
        print('Saved figure(s) can now be found here:   %s'%path)
    def web(fig, path, name):
        f.check_path(path) #from 'file.py'
        fig.write_html('%s%s.html'%(path,name))
        print('Saved figure(s) can now be found here:   %s'%path) 

def process(xax, int_raw, int_final, baseline, name, figType='jpeg'):
    '''
    Generates a 2 pane figure that shows raw spectra, baseline, and output spectra.

    Parameters
    ----------
    xax : np.array / float list
        X axis values (EX wavelengh)
    int_raw : np.array
        Spectra raw intensity
    int_final : np.array
        Spectral intensity  after processing
    baseline : np.array
        Baseline output, what is subtracted from the raw intensity
    name : string
        Name of file, will be used as the title in the resuling figure
    figType : 'jpeg', 'bmp', etc, optional
        Format type of the output figure. The default is 'jpeg'.

    Returns
    -------
    fig : Object
        2 panel figure that contains original spectra and resutling, processed, spectra

    '''
    #initalize figure
    fig=make_subplots(rows=2, cols=1,
                      shared_xaxes=True, shared_yaxes=True,
                      vertical_spacing=0.05,
                      x_title='Raman Shift (cm-1)', y_title='Intensity')
    #add traces
    #fig.append_trace(go.Scatter(x=xax, y=int_raw, name='raw'),row=1,col=1)
    #fig.append_trace(go.Scatter(x=xax, y=baseline, name='baseline'),row=1,col=1)
    fig.append_trace(go.Scatter(x=xax, y=int_final, name='output'),row=2,col=1)
    #adjust layout
    fig.update_layout(title_text=name,title_font_size=15,plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='white')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='white')
    return fig

def format(fig, title_size=30, axis_size=20, reverse_legend=True):
    fig.update_layout(title_font_size=title_size, plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=20,r=20,b=20,t=50),
                      legend=dict(yanchor='bottom',y=0))
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                     title_text='Relative Intensity', title_font_size=axis_size,
                     gridcolor='lightgrey')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black',
                     title_text='Raman Shift (cm-1)', title_font_size=axis_size,
                     gridcolor='lightgrey')
    if reverse_legend == True:
        fig.update_layout(legend_traceorder='reversed')
    elif reverse_legend == False:
        pass
    return fig
    