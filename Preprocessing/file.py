#!/usr/bin/m env python3
# -*- coding: utf-8 -*-
"""
Functions that find files, extract/edit file names (used for generating labels).

Created on Fri Jun 10 21:49:45 2022   -   @author: dunn
"""
import os

def check_path(dir_to_check): 
    '''
    Checks is directory exists. If not, directory will be created.
    If direct does exist, indicate so to user.

    Parameters
    ----------
    dir_to_check : TYPE
    
    check:
        True = directory exitsts
        False = directory does not exist

    Returns
    -------
    None.

    '''
    check=os.path.isdir(dir_to_check)
    #useful related code: for (root, dirs, files) in os.walk(indir):
    if check==True:
        print('\nOutput directory (%s) already exists.\n'%dir_to_check)
        print('%s contains %s.'%(dir_to_check, len(os.listdir(dir_to_check))))
    elif check == False:
        print('\nOutput directory (%s) does not currently exist, will be created for you.\n'%dir_to_check)
        os.mkdir(dir_to_check)
        print('\nThe following directory has been created for you: %s\n'%dir_to_check)
    else:
        print('\n***check path function error***\n')

class names():
    def find(path_to_dir, contains='', suffix='.csv'):
        '''
        Search a directory for all files, return a list of file names.
        
        Parameters
        ----------
        path_to_dir : String - path
            Path to the directory that contains desired files. 
        contains : string, optional
            A string to search for in files. 
            The default is '' (no specific charaters). 
        suffix : TYPE, optional
            File type to select. The default is '.csv'.

        Returns
        -------
        select : List
            List of file names that contain specified character(s) and file types.

        '''
        allfiles=[file for file in os.listdir(path_to_dir) if file.endswith(suffix)]
        select=[file for file in allfiles if contains in file]
        return select
    
    
    def edit(text, split_at_value, keep_location):
        #ADD DOC
        result=text.split(split_at_value)[keep_location]
        return result
        

class megaDF:
    '''
    This class contains functions related to dataframes which contain multiple samples/spectra.
    These functions have at least 3 columns, 'x', 'y', and 'name'.
    '''
    def splitAndSave(df, splitByCol, nameEnding, output_path): 
        #Split up the dataframe
        splits=list(df.groupby(splitByCol)) #Creates lift of tuples, each a dataframe for an averaged sample ramanShift+intensity
        i=0
        for splits[i] in splits:
            data=splits[i][1] #Access the dataframe portion of the tuple
            data.drop(splitByCol,axis=1,inplace=True) #Drop columns that you dont want saved to the final csv output
            fileName=splits[i][0] #Access the dataframe NAME portion of tuple
            format_fileName=fileName+nameEnding
            data.to_csv(output_path+'%s.csv'%format_fileName,index=False,header=False)
            i=i+1
        print('Each file should now be individually saved in %s.'%output_path)






