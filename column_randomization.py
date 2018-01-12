# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:50:13 2018

@author: JoanWang
"""

import pandas as pd
import sys
import numpy as np
import os


pd.options.mode.chained_assignment = None 

def randomize(filepath, index_col = None):
    '''
    Randomize column values of a file. Each column is randomized independently.
    
    Inputs:
        filepath (str): path to file to randomize; may be of type csv or txt
        index_col (str): optional name of column to use as index; will not be randomized
    
    Outputs:
        df (dataframe): dataframe representation of randomized data
        Output file will be generated, named as original file name + "_randomized"
    '''
    
    # Treat csv and txt differently
    filename, file_extension = os.path.splitext(filepath)
    if file_extension == '.csv':
        sep = ","
    elif file_extension == '.txt':
        sep = "\t"
        
    # Randomize each column
    df = pd.read_csv(filepath, sep = sep, index_col = index_col) 
    cols = df.columns
    for col in cols:
        print('... Randomizing column ' + col)
        df[col] = np.random.permutation(df[col])
        
    # Print to new csv or txt
    new_file = filename + '_randomized' + file_extension
    df.to_csv(new_file)   
    
    return df

if __name__ == "__main__":

    if len(sys.argv) == 2:
        file = sys.argv[1]
        df = randomize(file)
    elif len(sys.argv) == 3:
        file = sys.argv[1]
        index_col = sys.argv[2]
        df = randomize(file, index_col)    
    else:
        s = "usage: python3 {0} csv_or_txt_file index_col_name(optional)"
        s = s.format(sys.argv[0])
        print(s)
        sys.exit(0)

