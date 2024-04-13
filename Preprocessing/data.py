#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fun
ctions relating to data retrevial and processing.

Created on Sat Jun 11 00:34:03 2022

@author: dunn
"""
import numpy as np
#import pybaselines as py
from scipy.sparse import csc_matrix, eye, diags #smooth.whitaker
from scipy.sparse.linalg import spsolve #smooth.whitaker

def extract_col(df, i):
    '''
    Extracts an entire column of int/float data from dataframe,
    Returned as np.array.

    Parameters
    ----------
    df : DataFrame, int/float data column
        Dataframe containing column to extract as a list.
    i : integer
        Column location numerical value.

    Returns
    -------
    r : Array list, selected column data
    '''
    r=np.array(df.iloc[:,i])
    return r

class zap:
    def z_score(intensity):
        '''
        Basic Z-Score Function

        Parameters
        ----------
        intensity

        Returns
        -------
        zscores

        '''        
        mean=np.mean(intensity)
        std=np.std(intensity)
        zscores1=(intensity-mean)/std
        zscores=np.array(abs(zscores1))
        return(zscores)
    def mod_zscore(intensity):
        '''
        Parameters
        ----------
        intensity

        Returns
        -------
        mod_zscores

        '''
        median_int=np.median(intensity)
        mad_int=np.median([np.abs(intensity-median_int)])
        mod_z_scores1=0.6745*(intensity-median_int)/mad_int
        mod_z_scores=np.array(abs(mod_z_scores1))
        return mod_z_scores
    
    def WhitakerHayes_zscore(intensity, threshold):
        '''
        Whitaker-Hayes Function uses Intensity Modified Z-Scores

        Parameters
        ----------
        intensity : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        intensity_modified_zscores : TYPE
            DESCRIPTION.

        '''
        dist=0
        delta_intensity=[]
        for i in np.arange(len(intensity)-1):
            dist=intensity[i+1]-intensity[i]
            delta_intensity.append(dist)
        delta_int=np.array(delta_intensity)
        
        #Run the delta_int through MAD Z-Score Function
        intensity_modified_zscores=np.array(np.abs(zap.mod_zscore(delta_int)))
        return intensity_modified_zscores
    
    def detect_spike(z, threshold):
        '''
        #Add threshold parameter and use in zap function, right now this is redundant (6/2022)
        
        It detects spikes, or sudden, rapid changes in a spectral intensity.
        #potential edit: automatically generate spike graph
        
        Parameters
        ----------
        z : int/float
            Input z-score

        Returns
        -------
        spikes : Bool
            TRUE = spikes, FALSE = non-spikes.
            Potential edit: automatically generate spike graphs

        '''
        spikes=abs(abs(np.array(z))>threshold) #True assigned to spikes, False to non-spikes
        return (spikes)
    
    def fix(y, m=2, threshold=5):
        '''
        Replace spike intensity values with the average values that are not spikes in the selected range

        Parameters
        ----------
        y : Input intensity
            DESCRIPTION.
        m : int, selected range, optional
            Selction of points around the detected spike. The default is 2.
        threshold : TYPE, optional
            Binarization threshold. Increase value will increase spike detection sensitivity.
            I think.
            The default is 5.
            
        Returns
        -------
        y_out : float/array
            Average values that are around spikes in the selected range (~m)
        '''
        spikes=abs(np.array(zap.mod_zscore(np.diff(y))))>threshold
        y_out=y.copy() #Prevents overeyeride of input y
        for i in np.arange(len(spikes)):
          if spikes[i] !=0: #If there is a spike detected in position i
            w=np.arange(i-m, i+1+m) #Select 2m+1 points around the spike
            w2=w[spikes[w]==0] #From the interval, choose the ones which are not spikes
            if not w2.any(): #Empty array
                y_out[i]=np.mean(y[w]) #Average the values that are not spikes in the selected range        
            if w2.any(): #Normal array
                y_out[i]=np.mean(y[w2]) #Average the values that are not spikes in the selected range
        return y_out
    
class smooth:
    def Whittaker(x,w,lambda_,differences):
        '''
        Function formeraly (6/12/22), known as WhittakerSmooth
        Penalized least squares algorithm for background fitting.
        
        Parameters
        ----------
        x : float
            Input data (i.e. chromatogram of spectrum)
        w : binary masks
            value of the mask is 0 if a point belongs to peaks and one otherwise)
        lambda_ : TYPE
            Parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting backgrouand.
        differences : int
            Indicates the order of the difference of penalties.
        
        Returns
        -------
        The fitted background vector

        '''
        X=np.matrix(x)
        m=X.size
        i=np.arange(0,m)
        E=eye(m,format='csc')
        D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
        W=diags(w,0,shape=(m,m))
        A=csc_matrix(W+(lambda_*D.T*D))
        B=csc_matrix(W*X.T)
        background=spsolve(A,B)
        return np.array(background)   
    
class baseline:
    def airPLS(x, lambda_=100, itermax=15, porder=1):        
        '''
        Adaptive iteratively reweighted penalized lease squares for baseline fitting

        Parameters
        ----------
        x : float
            Input data (i.e. chromatogram of spectrum)
        lambda_ : TYPE
            Parameter that can be adjusted by user. The larger lambda is, the smoother the resulting background, z.
            Original value = 100.
        itermax : TYPE
            Original value = 15.
        porder : TYPE
            Adaptive iteratively reweighted penalized least squares for baseline fitting.
            Original value = 1.

        Returns
        -------
        z : TYPE
            DESCRIPTION.

        '''
        m=x.shape[0]
        w=np.ones(m)
        for i in range(1,itermax+1):
            z=smooth.Whittaker(x,w,lambda_, porder)
            d=x-z
            dssn=np.abs(d[d<0].sum())
            if(dssn<0.001*(abs(x)).sum() or i==itermax):
                if(i==itermax): print('WARING max iteration reached!')
                break
            w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
            w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
            w[0]=np.exp(i*(d[d<0]).max()/dssn)
            w[-1]=w[0]
        return z
    

